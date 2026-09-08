"""Semantic retrieval tier - CLIP embeddings over candidate cell chips.

Why a *delta* score rather than a similarity score
--------------------------------------------------
Asking "does this chip look like new construction?" is the wrong question: half
the cells in an Indian AOI already contain buildings, so absolute similarity
just ranks pre-existing cities. The question that matters is whether the place
*became* more like the query, so every candidate is embedded twice - once from
the pre-break composite, once from the post-break composite - and scored on

    delta = cos(after, query) - cos(before, query)

which is positive only where the scene moved toward the description. The chip
pair is already produced for the evidence panel, so this costs no extra I/O.

Loading is lazy and every failure is non-fatal: on a machine that cannot spare
the ~1 GB this needs, the pipeline falls back to the rule-based parser and says
so, rather than refusing to run.
"""
from __future__ import annotations

import os
import threading
import numpy as np

MODEL_NAME = "ViT-B-32"
FALLBACK_WEIGHTS = "laion2b_s34b_b79k"
REMOTECLIP_REPO = "chendelong/RemoteCLIP"
REMOTECLIP_FILE = "RemoteCLIP-ViT-B-32.pt"

# "auto" (use CLIP when it loads), "clip" (require it), "rules" (never load)
MODE = os.environ.get("GEOSENSE_SEMANTIC", "auto").lower()
THREADS = int(os.environ.get("GEOSENSE_TORCH_THREADS", "2"))

_lock = threading.Lock()
_state: dict = {"tried": False, "model": None, "preprocess": None,
                "tokenizer": None, "backend": None, "error": None}

# A short description of what each change signature looks like from orbit,
# ensembled with the user's own words so the query still drives retrieval.
# Each change class carries a natural-language description (what the model
# matches a query against) and the physical direction its indices move (how the
# detector then measures it). The description is the semantic half; the targets
# are physics, not heuristics - vegetation removal *does* lower NDVI.
CHANGE_CLASSES: dict[str, dict] = {
    "built_up": {
        "label": "New construction / built-up",
        "desc": "new buildings, construction sites and built-up expansion on "
                "cleared ground; new housing, factories or urban development",
        "targets": {"ndvi": -1.0, "ndbi": +1.0, "bsi": +0.6},
        "sar": "stabilise"},
    "veg_loss": {
        "label": "Vegetation / forest loss",
        "desc": "deforestation, forest clearing, felled trees, logging and "
                "loss of green canopy leaving bare soil",
        "targets": {"ndvi": -1.2, "bsi": +0.5},
        "requires": {"pre_ndvi_min": 0.35},
        "sar": "destabilise"},
    "landslide": {
        "label": "Landslide / slope failure",
        "desc": "a landslide scar on a hillside, slope failure, debris flow, "
                "collapsed earth and exposed soil on steep terrain after rain",
        "targets": {"ndvi": -1.2, "bsi": +1.0},
        "requires": {"pre_ndvi_min": 0.25},
        "sar": "destabilise"},
    "damage": {
        "label": "Structural damage / destruction",
        "desc": "destroyed or damaged buildings, rubble and debris where "
                "structures used to stand, demolition and collapsed roofs",
        "targets": {"ndbi": -0.6, "bsi": +1.0, "ndvi": -0.4},
        "sar": "destabilise"},
    "bare_soil": {
        "label": "Quarrying / bare ground",
        "desc": "an open quarry, mine, sand extraction or excavated bare "
                "earth exposed where ground was previously covered",
        "targets": {"bsi": +1.0, "ndvi": -0.7, "ndbi": +0.4},
        "sar": "none"},
    "flood": {
        "label": "Flooding / inundation",
        "desc": "farmland and settlements submerged under flood water after "
                "heavy monsoon rain, standing water covering land",
        "targets": {"mndwi": +1.2, "ndvi": -0.5},
        "sar": "destabilise"},
    "water_loss": {
        "label": "Water body contraction",
        "desc": "a lake, tank or reservoir drying up, shrinking water extent "
                "and exposed shoreline or dry bed",
        "targets": {"mndwi": -1.2},
        "requires": {"pre_mndwi_min": -0.1},
        "sar": "none"},
    "veg_gain": {
        "label": "Vegetation gain / regrowth",
        "desc": "vegetation growing back, trees returning, new plantation "
                "and land turning green again after being bare",
        "targets": {"ndvi": +1.2},
        "sar": "none"},
    "linear": {
        "label": "New road / linear corridor",
        "desc": "a new road, highway, track or canal cutting a straight "
                "linear corridor across land",
        "targets": {"ndvi": -0.8, "ndbi": +1.0, "bsi": +0.6},
        "shape": "linear",
        "sar": "stabilise"},
}

# kept for the CLIP delta prompt used during re-ranking
SIGNATURE_PROMPTS = {k: "a satellite image of " + v["desc"].split(";")[0]
                     for k, v in CHANGE_CLASSES.items()}
SIGNATURE_PROMPTS["any"] = "a satellite image of changed ground"

_class_vecs: dict = {"keys": None, "matrix": None, "backend": None}
_text_state: dict = {"tried": False, "model": None, "error": None}

TEXT_MODEL = os.environ.get("GEOSENSE_TEXT_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")


def load_text_model():
    """Sentence-embedding model for query understanding.

    CLIP is trained to align text with *images*, not text with text, so using
    its text tower to compare a query against class descriptions is off-label
    and measurably weaker: on an 11-query check it scored 9/11 with margins of
    0.03-0.15, against 10/11 and margins of 0.19-0.45 here. Wide margins are
    what make "uncertain" a usable signal rather than noise, so query
    classification uses a model built for the job and falls back to CLIP.
    """
    if _text_state["tried"]:
        return _text_state["model"]
    with _lock:
        _text_state["tried"] = True
        try:
            from sentence_transformers import SentenceTransformer
            _text_state["model"] = SentenceTransformer(TEXT_MODEL)
        except Exception as ex:
            _text_state["error"] = f"{type(ex).__name__}: {str(ex)[:140]}"
    return _text_state["model"]


def _class_matrix():
    """Embed every class description once, with the best model available."""
    if _class_vecs["matrix"] is not None:
        return _class_vecs["keys"], _class_vecs["matrix"], _class_vecs["backend"]
    import numpy as np
    keys = list(CHANGE_CLASSES)
    docs = [f"{CHANGE_CLASSES[k]['label']}. {CHANGE_CLASSES[k]['desc']}"
            for k in keys]

    tm = load_text_model()
    if tm is not None:
        M = np.asarray(tm.encode(docs, normalize_embeddings=True))
        _class_vecs.update(keys=keys, matrix=M, backend=TEXT_MODEL.split("/")[-1])
        return keys, M, _class_vecs["backend"]

    if load():                                   # CLIP text tower fallback
        vecs = [encode_text(d) for d in docs]
        if not any(v is None for v in vecs):
            M = np.stack(vecs)
            _class_vecs.update(keys=keys, matrix=M,
                               backend=f"{MODEL_NAME} text tower (fallback)")
            return keys, M, _class_vecs["backend"]
    return None, None, None


def embed_query(text: str):
    """Query vector from whichever model backs the class matrix."""
    tm = load_text_model()
    if tm is not None:
        import numpy as np
        return np.asarray(tm.encode(text, normalize_embeddings=True))
    return encode_text(text)


def classify_query(text: str, margin: float = 0.05) -> dict | None:
    """Map free text onto a change class with a model, not keywords.

    Keyword matching fails silently and confidently: "landslide and built up
    damage" matched the substring "built" and searched for NEW CONSTRUCTION -
    the opposite of what was asked - without ever saying so. Comparing
    embeddings instead resolves on meaning, and the margin to the runner-up
    gives an honest confidence the caller can act on.

    Returns None when no model is available, so the caller can fall back.
    """
    import numpy as np
    keys, matrix, backend = _class_matrix()
    if keys is None:
        return None
    q = embed_query(text)
    if q is None:
        return None

    sims = matrix @ np.asarray(q)
    order = np.argsort(-sims)
    ranked = [{"class": keys[i], "label": CHANGE_CLASSES[keys[i]]["label"],
               "score": round(float(sims[i]), 4)} for i in order]
    gap = ranked[0]["score"] - ranked[1]["score"]
    return {"class": ranked[0]["class"], "label": ranked[0]["label"],
            "score": ranked[0]["score"], "margin": round(float(gap), 4),
            "confident": bool(gap >= margin),
            "ranked": ranked[:4], "backend": backend}


def status() -> dict:
    return {"mode": MODE, "backend": _state["backend"],
            "loaded": _state["model"] is not None, "error": _state["error"]}


def load() -> bool:
    """Load CLIP once. Returns False (never raises) if it cannot be used."""
    if MODE == "rules":
        _state["error"] = "disabled by GEOSENSE_SEMANTIC=rules"
        return False
    with _lock:
        if _state["tried"]:
            return _state["model"] is not None
        _state["tried"] = True
        try:
            import torch
            import open_clip
            torch.set_num_threads(THREADS)

            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=FALLBACK_WEIGHTS)
            backend = f"CLIP {MODEL_NAME} ({FALLBACK_WEIGHTS})"

            # Prefer RemoteCLIP: the same architecture fine-tuned on remote
            # sensing image-text pairs, so orbital vocabulary actually lands.
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(REMOTECLIP_REPO, REMOTECLIP_FILE)
                sd = torch.load(path, map_location="cpu")
                model.load_state_dict(sd, strict=False)
                backend = f"RemoteCLIP {MODEL_NAME}"
            except Exception:
                pass                       # generic CLIP is a fine fallback

            model.eval()
            _state.update(model=model, preprocess=preprocess,
                          tokenizer=open_clip.get_tokenizer(MODEL_NAME),
                          backend=backend, error=None)
            return True
        except Exception as ex:            # missing torch, OOM, no network
            _state["error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
            return False


def _to_pil(arr):
    """A float reflectance chip (H,W,3) -> 8-bit PIL image, same stretch as
    the evidence chips so what CLIP sees is what the analyst sees."""
    from PIL import Image
    a = np.nan_to_num(np.asarray(arr, dtype="float32"), nan=0.0)
    a = np.clip(a / 0.30, 0, 1) ** (1 / 1.5)
    return Image.fromarray((a * 255).astype("uint8")).resize((224, 224),
                                                            Image.BILINEAR)


def encode_images(chips, batch=8) -> np.ndarray | None:
    """chips: list of (H,W,3) reflectance arrays -> L2-normalised (N,512)."""
    if not chips or not load():
        return None
    import torch
    pre, model = _state["preprocess"], _state["model"]
    out = []
    with torch.no_grad():
        for i in range(0, len(chips), batch):
            t = torch.stack([pre(_to_pil(c)) for c in chips[i:i + batch]])
            f = model.encode_image(t)
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype("float32")


def encode_text(*texts) -> np.ndarray | None:
    """Ensemble several phrasings into one L2-normalised query vector."""
    texts = [t for t in texts if t]
    if not texts or not load():
        return None
    import torch
    with torch.no_grad():
        f = _state["model"].encode_text(_state["tokenizer"](list(texts)))
        f = f / f.norm(dim=-1, keepdim=True)
        v = f.mean(0)
        v = v / v.norm()
    return v.cpu().numpy().astype("float32")


def query_vector(raw_text: str, signature: str) -> np.ndarray | None:
    return encode_text(raw_text, SIGNATURE_PROMPTS.get(signature))


def delta_scores(before_chips, after_chips, qvec, return_vectors=False):
    """cos(after, q) - cos(before, q) per candidate. Positive = moved toward
    the description.

    With `return_vectors` the post-change embeddings come back too, so they can
    be stored and searched later - the embedding is the expensive part and
    discarding it after one comparison wastes the whole point of having one.
    """
    if qvec is None:
        return (None, None) if return_vectors else None
    a = encode_images(after_chips)
    b = encode_images(before_chips)
    if a is None or b is None:
        return (None, None) if return_vectors else None
    scores = (a @ qvec) - (b @ qvec)
    return (scores, a, b) if return_vectors else scores


def probe() -> dict:
    """Cheap availability check - does NOT load the 1.5 GB model.

    Reports whether the libraries are installed and whether the weights are
    already in the HuggingFace cache, so the UI can distinguish "not installed"
    from "installed but switched off" from "downloading on first use". Anything
    that answers this by importing torch would cost ~200 MB just to draw a
    label, so it only inspects the filesystem and the import machinery.
    """
    import importlib.util as iu
    installed = all(iu.find_spec(m) is not None for m in ("torch", "open_clip"))

    cached = False
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(REMOTECLIP_REPO, REMOTECLIP_FILE)
        cached = isinstance(hit, str) and os.path.exists(hit)
    except Exception:
        pass
    if not cached:                       # generic CLIP weights land in ~/.cache
        home = os.path.expanduser("~/.cache/huggingface/hub")
        cached = os.path.isdir(home) and any(
            "CLIP-ViT-B-32" in d or "RemoteCLIP" in d
            for d in os.listdir(home)) if os.path.isdir(home) else False

    if MODE == "rules":
        state = "disabled"
    elif not installed:
        state = "not installed"
    elif _state["model"] is not None:
        state = "loaded"
    elif cached:
        state = "ready"                  # weights on disk, loads in seconds
    else:
        state = "will download (~600 MB)"

    return {"installed": installed, "weights_cached": cached,
            "loaded": _state["model"] is not None,
            "state": state,
            "model": f"RemoteCLIP {MODEL_NAME}" if installed else None}


def classify_change(before_vecs, after_vecs):
    """What kind of change is this, judged from the imagery itself?

    The query classifier reads the *question*; this reads the *answer*. Each
    class prompt is scored the same way the re-ranker scores a query - by how
    much the scene moved toward it - so a place that merely already looks like
    a city does not win the "construction" class.

    Returns one dict per candidate, or None if the model is unavailable.
    """
    import numpy as np
    if before_vecs is None or after_vecs is None or not load():
        return None
    keys = list(CHANGE_CLASSES)
    prompts = [encode_text(SIGNATURE_PROMPTS[k], CHANGE_CLASSES[k]["label"])
               for k in keys]
    if any(v is None for v in prompts):
        return None
    P = np.stack(prompts)

    out = []
    deltas = (np.asarray(after_vecs) @ P.T) - (np.asarray(before_vecs) @ P.T)
    for row in deltas:
        order = np.argsort(-row)
        ranked = [{"class": keys[i], "label": CHANGE_CLASSES[keys[i]]["label"],
                   "score": round(float(row[i]), 4)} for i in order[:3]]
        margin = float(row[order[0]] - row[order[1]])
        out.append({"predicted": ranked[0]["class"], "label": ranked[0]["label"],
                    "score": ranked[0]["score"], "margin": round(margin, 4),
                    "confident": bool(margin >= 0.01), "ranked": ranked,
                    "source": _state["backend"]})
    return out
