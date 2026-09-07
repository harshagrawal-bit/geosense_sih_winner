"""Prioritise tier: score cells against the query signature, fuse the evidence
streams with Reciprocal Rank Fusion, and attach calibrated confidence.

The confidence is a conformal p-value computed against the scene's own null
distribution. Most cells in any AOI are unchanged, so the empirical
distribution of scores across cells is a usable calibration set: for a cell
with score s,

    p = (1 + #{cells with score >= s}) / (n_cells + 1)

is a valid conformal p-value under exchangeability of the unchanged cells.
Benjamini-Hochberg on those p-values then controls the false discovery rate at
the requested alpha, so "6 detections at alpha=0.10" is a statement with
content rather than a slogan.
"""
from __future__ import annotations
import numpy as np

RRF_K = 60          # standard RRF damping constant


def signature_score(breaks, targets, requires, pre_means):
    """Weighted agreement between each cell's breaks and the query signature."""
    keys = [k for k in targets if k in breaks]
    if not keys:
        return None, None
    shape = next(iter(breaks.values()))["z"].shape
    score = np.zeros(shape)
    total_w = 0.0
    detail = {}

    for k in keys:
        w = targets[k]
        z = np.nan_to_num(breaks[k]["z"], nan=0.0)
        mag = np.nan_to_num(breaks[k]["magnitude"], nan=0.0)
        if w == 0.0:                      # "any change" mode: magnitude only
            contrib = np.abs(z)
            total_w += 1.0
        else:
            # reward change in the expected direction, penalise the opposite
            contrib = np.sign(w) * z
            total_w += abs(w)
            contrib = contrib * abs(w)
        detail[k] = {"z": z, "mag": mag}
        score += contrib

    score /= max(total_w, 1e-6)

    # hard priors: e.g. forest loss requires there to have been forest
    if requires.get("pre_ndvi_min") is not None and "ndvi" in pre_means:
        score = np.where(pre_means["ndvi"] >= requires["pre_ndvi_min"],
                         score, score - 3.0)
    if requires.get("pre_mndwi_min") is not None and "mndwi" in pre_means:
        score = np.where(pre_means["mndwi"] >= requires["pre_mndwi_min"],
                         score, score - 3.0)
    return score, detail


def context_bonus(ctx, pre_means):
    """Spatial context from the imagery itself, not from a gazetteer."""
    b = np.zeros(next(iter(pre_means.values())).shape)
    if ctx.get("near_water") and "mndwi" in pre_means:
        w = pre_means["mndwi"]
        near = _dilate(w > -0.15, 2)
        b += np.where(near, 0.8, -0.4)
    if ctx.get("near_forest") and "ndvi" in pre_means:
        b += np.where(_dilate(pre_means["ndvi"] > 0.45, 1), 0.6, -0.3)
    if ctx.get("near_urban") and "ndbi" in pre_means:
        b += np.where(_dilate(pre_means["ndbi"] > 0.0, 1), 0.5, -0.2)
    return b


def _dilate(mask, r=1):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def linear_bonus(z):
    """Cheap elongation prior: a road lights up cells in a line, so a cell whose
    strong neighbours are collinear scores above an isolated blob."""
    s = np.abs(np.nan_to_num(z))
    h = np.roll(s, 1, 1) + np.roll(s, -1, 1)
    v = np.roll(s, 1, 0) + np.roll(s, -1, 0)
    return np.abs(h - v) / (h + v + 1e-6)


def rrf(*rank_arrays, k=RRF_K):
    """Reciprocal Rank Fusion over several evidence streams."""
    fused = np.zeros_like(rank_arrays[0], dtype=float)
    for r in rank_arrays:
        fused += 1.0 / (k + r)
    return fused


def ranks_of(score, descending=True):
    flat = score.ravel()
    order = np.argsort(-flat if descending else flat, kind="stable")
    r = np.empty_like(order)
    r[order] = np.arange(len(flat))
    return r.reshape(score.shape).astype(float)


def conformal(score):
    """Conformal p-value per cell against the in-scene null distribution."""
    flat = np.nan_to_num(score.ravel(), nan=-1e9)
    n = flat.size
    order = np.argsort(-flat, kind="stable")
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1)
    p = (1.0 + ranks - 1.0) / (n + 1.0)         # #{>= s} = rank
    return p.reshape(score.shape)


def bh_threshold(p, alpha=0.10):
    """Benjamini-Hochberg: largest p_(i) with p_(i) <= alpha*i/n."""
    flat = np.sort(p.ravel())
    n = flat.size
    ok = flat <= alpha * np.arange(1, n + 1) / n
    return float(flat[ok][-1]) if ok.any() else 0.0
