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
SIGNATURE_PROMPTS = {
    "built_up":  "a satellite image of new buildings and construction on cleared ground",
    "veg_loss":  "a satellite image of cleared forest, cut trees and exposed bare soil",
    "bare_soil": "a satellite image of an open quarry, mine or excavated bare earth",
    "flood":     "a satellite image of farmland submerged under flood water",
    "water_loss": "a satellite image of a dried up lake bed with exposed shoreline",
    "veg_gain":  "a satellite image of dense green vegetation and new plantation",
    "linear":    "a satellite image of a new road cutting a straight line through land",
    "any":       "a satellite image of changed ground",
}


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


def delta_scores(before_chips, after_chips, qvec) -> np.ndarray | None:
    """cos(after, q) - cos(before, q) per candidate. Positive = moved toward
    the description."""
    if qvec is None:
        return None
    a = encode_images(after_chips)
    b = encode_images(before_chips)
    if a is None or b is None:
        return None
    return (a @ qvec) - (b @ qvec)
