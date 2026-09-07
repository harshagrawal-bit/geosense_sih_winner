"""Natural-language query -> physical change signature.

This is the Discover tier. The CLIP/GeoRSCLIP embedding backend is not enabled
in this build (see README), so a query is resolved into the spectral signature
the described change actually produces on the ground. Every rule is inspectable
and the UI shows which ones fired, which is worth more in a review workflow
than an opaque similarity score.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List

# index -> expected direction of the break, and its weight in the score
SIGNATURES: Dict[str, Dict[str, Any]] = {
    "built_up": {
        "label": "New construction / built-up",
        "terms": ["construction", "building", "built", "built-up", "urban",
                  "settlement", "encroach", "colony", "house", "housing",
                  "development", "concrete", "structure", "township"],
        "targets": {"ndvi": -1.0, "ndbi": +1.0, "bsi": +0.6},
        "sar": "stabilise",
    },
    "veg_loss": {
        "label": "Vegetation / forest loss",
        "terms": ["deforest", "forest loss", "clearing", "cleared", "logging",
                  "felling", "tree", "canopy", "vegetation loss", "cut down"],
        "targets": {"ndvi": -1.2, "bsi": +0.5},
        "requires": {"pre_ndvi_min": 0.35},
        "sar": "destabilise",
    },
    "bare_soil": {
        "label": "Quarrying / bare ground",
        "terms": ["quarry", "quarrying", "mine", "mining", "excavat",
                  "bare", "soil", "sand", "borrow pit", "dump"],
        "targets": {"bsi": +1.0, "ndvi": -0.7, "ndbi": +0.4},
        "sar": "none",
    },
    "flood": {
        "label": "Flooding / inundation",
        "terms": ["flood", "inundat", "submerg", "waterlog", "deluge",
                  "standing water", "monsoon damage"],
        "targets": {"mndwi": +1.2, "ndvi": -0.5},
        "sar": "destabilise",
    },
    "water_loss": {
        "label": "Water body contraction",
        "terms": ["water loss", "loss of water", "loss of a water", "water body",
                  "dried", "drying", "dry up", "dried up", "shrink", "shrinking",
                  "receding", "lake", "tank", "reservoir", "pond", "wetland"],
        "targets": {"mndwi": -1.2},
        "requires": {"pre_mndwi_min": -0.1},
        "sar": "none",
    },
    "veg_gain": {
        "label": "Vegetation gain / regrowth",
        "terms": ["regrowth", "afforest", "plantation", "greening",
                  "restored", "vegetation gain", "replant"],
        "targets": {"ndvi": +1.2},
        "sar": "none",
    },
    "linear": {
        "label": "New road / linear corridor",
        "terms": ["road", "highway", "corridor", "track", "runway", "railway",
                  "canal", "pipeline", "alignment"],
        "targets": {"ndvi": -0.8, "ndbi": +1.0, "bsi": +0.6},
        "shape": "linear",
        "sar": "stabilise",
    },
}

CONTEXT = {
    "near_water": ["near a river", "near river", "riverside", "floodplain",
                   "near water", "along the river", "riverbank", "near a lake",
                   "near the coast", "waterfront"],
    "near_forest": ["forest edge", "in forest", "inside forest", "woodland",
                    "reserve", "sanctuary", "national park", "buffer"],
    "near_urban":  ["near city", "urban fringe", "peri-urban", "outskirts",
                    "near town", "city edge"],
}

MONTHS = {"month": 1, "months": 1, "year": 12, "years": 12,
          "quarter": 3, "quarters": 3, "week": 0.25, "weeks": 0.25}


def parse(text: str, use_model: bool = True) -> Dict[str, Any]:
    """Resolve a query to a change signature.

    Classification is done by a sentence-embedding model; the keyword table
    below survives only as an offline fallback for when no model is staged.
    Context and time-window extraction stay lexical because they are literal
    facts in the text ("near a river", "last 18 months"), not semantics.
    """
    q = (text or "").lower().strip()

    model_pick = None
    if use_model:
        try:
            from . import semantic as _sem
            model_pick = _sem.classify_query(text)
        except Exception:
            model_pick = None

    scores = {}
    hits: Dict[str, List[str]] = {}
    for key, sig in SIGNATURES.items():
        matched = [t for t in sig["terms"] if t in q]
        if matched:
            scores[key] = len(matched) + 0.5 * max(len(t) for t in matched) / 10
            hits[key] = matched

    if model_pick is not None:
        from .semantic import CHANGE_CLASSES
        chosen = model_pick["class"]
        sig = CHANGE_CLASSES[chosen]
    elif not scores:                    # no model, no keyword -> monitor all
        chosen = "any"
        sig = {"label": "Any significant change", "targets":
               {"ndvi": 0.0, "ndbi": 0.0, "mndwi": 0.0, "bsi": 0.0},
               "sar": "none"}
    else:
        chosen = max(scores, key=scores.get)
        sig = SIGNATURES[chosen]

    ctx = {k: any(p in q for p in pats) for k, pats in CONTEXT.items()}

    months = None
    m = re.search(r"(?:last|past|previous)\s+(\d+)\s*(month|months|year|years|"
                  r"quarter|quarters|week|weeks)", q)
    if m:
        months = max(3, int(round(int(m.group(1)) * MONTHS[m.group(2)])))
    elif "last year" in q or "past year" in q:
        months = 12

    explain = []
    if model_pick is not None:
        explain.append(
            f"Classified as '{model_pick['label']}' by {model_pick['backend']} "
            f"(score {model_pick['score']:.3f}, margin {model_pick['margin']:.3f}).")
        if not model_pick["confident"]:
            runner = model_pick["ranked"][1]
            explain.append(
                f"LOW CONFIDENCE - '{runner['label']}' scored almost as high. "
                f"Treat the ranking as provisional.")
        for idx, w in sig["targets"].items():
            explain.append(f"Expect {idx.upper()} to "
                           f"{'rise' if w > 0 else 'fall'} (weight {abs(w):.1f}).")
    elif chosen == "any":
        explain.append("No signature keyword matched - ranking by overall "
                       "change magnitude across all indices.")
    else:
        explain.append(f"Matched signature '{sig['label']}' on "
                       f"{', '.join(repr(t) for t in hits[chosen])}.")
        for idx, w in sig["targets"].items():
            explain.append(f"Expect {idx.upper()} to "
                           f"{'rise' if w > 0 else 'fall'} (weight {abs(w):.1f}).")
    for k, on in ctx.items():
        if on:
            explain.append(f"Context filter: {k.replace('_', ' ')}.")
    if months:
        explain.append(f"Time window narrowed to the last {months} months.")

    return {"raw": text, "signature": chosen, "label": sig["label"],
            "classifier": (model_pick or {}).get("backend", "keyword fallback"),
            "confidence": (model_pick or {}).get("score"),
            "margin": (model_pick or {}).get("margin"),
            "confident": (model_pick or {}).get("confident", True),
            "alternatives": (model_pick or {}).get("ranked", [])[1:4],
            "targets": sig["targets"], "requires": sig.get("requires", {}),
            "shape": sig.get("shape"), "sar_expect": sig.get("sar", "none"),
            "context": ctx, "months": months, "explain": explain,
            "matched_terms": hits.get(chosen, [])}
