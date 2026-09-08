"""Physical plausibility checks on a detection.

The change detector answers "did the composite move in the direction the query
implies?" - a question about *direction*. It never asks whether the resulting
*state* is physically consistent with the thing that was requested, and that
gap produced a real false positive: a flood query over the Thar desert
certified a cell whose wetness rose slightly while its bare-soil and built-up
indices also rose. Under real water those two fall, because water absorbs
shortwave infrared almost completely. It was dry ground getting drier.

These are not classification heuristics. They are physical constraints on what
the imagery must look like *after* the change for the requested class to be
possible at all, and they are used to warn, never to silently discard.
"""
from __future__ import annotations

# Levels an index must reach, or must not exceed, for a class to be plausible.
# Water sits above 0 in MNDWI; anything well below is not water.
GATES: dict[str, list[dict]] = {
    "flood": [
        {"index": "mndwi", "when": "post", "min": -0.15,
         "why": "standing water needs MNDWI near or above 0; well below that is dry ground"},
        {"index": "ndbi", "when": "delta", "max": 0.02,
         "why": "water absorbs shortwave infrared, so NDBI falls under flooding - a rise contradicts it"},
        {"index": "bsi", "when": "delta", "max": 0.05,
         "why": "bare-soil index rises as ground dries, the opposite of inundation"},
    ],
    "water_loss": [
        {"index": "mndwi", "when": "pre", "min": -0.15,
         "why": "there must have been water present before it could be lost"},
    ],
    "veg_loss": [
        {"index": "ndvi", "when": "pre", "min": 0.30,
         "why": "vegetation must have been there before it could be cleared"},
    ],
    "landslide": [
        {"index": "ndvi", "when": "pre", "min": 0.20,
         "why": "a slope failure strips existing cover; bare ground cannot slip visibly"},
        {"index": "bsi", "when": "delta", "min": 0.01,
         "why": "a landslide exposes soil, so the bare-soil index should rise"},
    ],
    "built_up": [
        {"index": "ndbi", "when": "delta", "min": -0.01,
         "why": "new hard surfaces raise the built-up index"},
    ],
    "veg_gain": [
        {"index": "ndvi", "when": "delta", "min": 0.01,
         "why": "regrowth must raise NDVI"},
    ],
}


def check(change_class: str, pre: dict, post: dict, delta: dict) -> list[dict]:
    """Return the gates this detection fails for the requested class."""
    failures = []
    for gate in GATES.get(change_class, []):
        idx = gate["index"]
        src = {"pre": pre, "post": post, "delta": delta}[gate["when"]]
        val = src.get(idx)
        if val is None:
            continue
        lo, hi = gate.get("min"), gate.get("max")
        bad = (lo is not None and val < lo) or (hi is not None and val > hi)
        if bad:
            bound = f"≥ {lo}" if lo is not None else f"≤ {hi}"
            failures.append({
                "index": idx.upper(), "when": gate["when"],
                "value": round(float(val), 3), "required": bound,
                "why": gate["why"],
                "text": f"{idx.upper()} {gate['when']} = {val:+.3f}, "
                        f"needs {bound} — {gate['why']}"})
    return failures


def verdict(requested: str, predicted: str | None, failures: list[dict]) -> dict:
    """Combine the physical gates with what the image model saw."""
    agrees = predicted is None or predicted == requested
    if failures and not agrees:
        level, msg = "contradicted", (
            f"The evidence does not support '{requested}'. It looks like "
            f"'{predicted}', and {len(failures)} physical check(s) fail.")
    elif failures:
        level, msg = "implausible", (
            f"{len(failures)} physical check(s) fail for '{requested}'.")
    elif not agrees:
        level, msg = "reclassified", (
            f"Physically possible, but the imagery looks more like "
            f"'{predicted}' than '{requested}'.")
    else:
        level, msg = "supported", "Evidence is consistent with the query."
    return {"level": level, "message": msg, "agrees": bool(agrees),
            "n_failures": len(failures)}
