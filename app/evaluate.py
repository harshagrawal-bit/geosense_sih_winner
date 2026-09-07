"""Semi-synthetic validation: inject known change into real imagery.

There is no public benchmark that matches what this detector consumes. The
standard change-detection sets (OSCD, LEVIR-CD) are *bi-temporal* - two dates
and a mask - whereas this pipeline needs a 15+ date time series to fit a
seasonal model. Scoring against them would test a different algorithm.

So we validate the way a radar engineer validates a receiver: take real signal
(a genuine Sentinel-2 stack over a real AOI, with its real noise, real
phenology and real cloud gaps), inject a change of known size at a known place
and time, and measure how often the detector finds it. Sweeping the size gives
a power curve - the minimum change the system can actually see. Injecting
nothing measures the false-positive rate, which is the only honest way to check
that the Benjamini-Hochberg guarantee holds in practice rather than on paper.

Every function below calls the same `change.*` and `rank.*` routines the live
pipeline calls, so this measures the shipped detector, not a copy of it.
"""
from __future__ import annotations

import json
import os
import numpy as np

from . import catalog, change, raster, rank
from .config import CACHE_DIR

WANT = ("red", "nir", "green", "swir16", "blue")

# A construction event does not move one index in isolation: vegetation falls,
# built-up rises, bare soil rises. Injecting that coupled signature is a fairer
# test than nudging NDVI alone.
SIGNATURE = {"ndvi": -1.0, "ndbi": +0.8, "bsi": +0.5}
TARGETS = {"ndvi": -1.0, "ndbi": +1.0, "bsi": +0.6}      # what query.py emits


# --------------------------------------------------------------- real data --
def fetch_cube(bbox, start, end, cloud=45, max_scenes=26, cache=None):
    """Download a real index time series once and cache it to disk."""
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return list(d["dates"]), {k: d[k] for k in d.files if k != "dates"}

    items = catalog.search("sentinel-2-l2a", bbox, start, end,
                           cloud=cloud, limit=max_scenes)
    scenes = raster.optical_stack(items, bbox, want=WANT)
    for min_valid in (0.35, 0.20, 0.10):
        dates, st, _ = raster.assemble(scenes, WANT, min_valid)
        if len(dates) >= 10:
            break
    if len(dates) < 10:
        raise RuntimeError(f"only {len(dates)} usable dates over this AOI")

    ix = raster.indices(st)
    cells = {k: change.to_cells(v) for k, v in ix.items()}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, dates=np.array(dates), **cells)
    return dates, cells


# ---------------------------------------------------------------- injection --
def inject(cells, break_idx, blocks, magnitude):
    """Add a step change of known size to known cells from `break_idx` on."""
    out = {k: v.copy() for k, v in cells.items()}
    for (j, i) in blocks:
        for k, w in SIGNATURE.items():
            if k in out:
                out[k][break_idx:, j, i] += w * magnitude
    return out


def pick_blocks(ny, nx, n_blocks, block, rng, margin=1):
    """Place non-overlapping square blocks, away from the grid edge."""
    chosen, taken = [], set()
    for _ in range(400):
        if len(chosen) >= n_blocks:
            break
        j = rng.integers(margin, ny - block - margin)
        i = rng.integers(margin, nx - block - margin)
        cand = {(j + dj, i + di) for dj in range(block) for di in range(block)}
        if taken & {(a + dj, b + di) for (a, b) in cand
                    for dj in (-1, 0, 1) for di in (-1, 0, 1)}:
            continue
        chosen.append(sorted(cand))
        taken |= cand
    return chosen


# ----------------------------------------------------------------- detector --
def detect(cells, dates, alpha=0.10, n_perm=120, seed=7):
    """Exactly the CUE stage of pipeline.run, on an in-memory cube."""
    ny, nx = next(iter(cells.values())).shape[1:]
    keep_idx, _ = change.select_dates(cells["ndvi"], dates)
    if len(keep_idx) < 8:
        raise RuntimeError("not enough usable dates")
    kept = [dates[i] for i in keep_idx]

    resid = {k: change.deseasonalize(change.fill_gaps(c[keep_idx]), kept)
             for k, c in cells.items()}
    w = {k: v for k, v in TARGETS.items() if k in resid}
    tot = sum(abs(v) for v in w.values())
    comp = sum((v / tot) * resid[k] for k, v in w.items())

    z, _, _ = change.multiscale_scan(comp, ny, nx, sign=1)
    null = change.multiscale_null(comp, ny, nx, sign=1, n_perm=n_perm,
                                  seed=seed, sample=160)
    p = change.conformal_p(z, null, sign=1).reshape(ny, nx)
    thr = rank.bh_threshold(p, alpha)
    return (p <= thr), p, z.reshape(ny, nx), keep_idx


# ------------------------------------------------------------------ scoring --
def score(detected, truth_cells, ny, nx, tolerance=1):
    """TP/FP/FN with a spatial tolerance.

    The detector applies a 3x3 matched filter, so a real change legitimately
    lights up the cells adjacent to it. Counting those as false alarms would
    punish the filter for working. `tolerance=1` accepts a hit within one cell
    of injected ground truth; strict (tolerance=0) is reported alongside.
    """
    truth = set(truth_cells)
    halo = {(j + dj, i + di) for (j, i) in truth
            for dj in range(-tolerance, tolerance + 1)
            for di in range(-tolerance, tolerance + 1)}
    det = {(int(j), int(i)) for j, i in np.argwhere(detected)}

    tp = len(det & halo)
    fp = len(det - halo)
    found = len({c for c in truth if any((c[0] + dj, c[1] + di) in det
                for dj in range(-tolerance, tolerance + 1)
                for di in range(-tolerance, tolerance + 1))})
    return {"detected": len(det), "tp": tp, "fp": fp,
            "truth": len(truth), "truth_found": found,
            "precision": tp / len(det) if det else float("nan"),
            "recall": found / len(truth) if truth else float("nan"),
            "fdr": fp / len(det) if det else 0.0}


# -------------------------------------------------------------------- sweep --
def sweep(cells, dates, magnitudes, repeats=5, alpha=0.10,
          n_blocks=3, block=2, base_seed=0, log=print):
    """Power curve: detection rate as a function of injected change size."""
    ny, nx = next(iter(cells.values())).shape[1:]
    n_dates = len(dates)
    rows = []

    for m in magnitudes:
        agg = []
        for r in range(repeats):
            rng = np.random.default_rng(base_seed + r * 101 + int(m * 1000))
            blocks = pick_blocks(ny, nx, n_blocks, block, rng)
            truth = [c for b in blocks for c in b] if m > 0 else []
            bi = int(rng.integers(max(n_dates // 3, 4), n_dates - 4)) \
                if n_dates > 10 else n_dates // 2
            cube = inject(cells, bi, truth, m) if m > 0 else cells
            det, p, z, _ = detect(cube, dates, alpha=alpha, seed=7 + r)
            agg.append(score(det, truth, ny, nx))
        row = {"magnitude": round(float(m), 3),
               "recall": float(np.nanmean([a["recall"] for a in agg])),
               "precision": float(np.nanmean([a["precision"] for a in agg])),
               "fdr": float(np.mean([a["fdr"] for a in agg])),
               "detected": float(np.mean([a["detected"] for a in agg])),
               "repeats": repeats}
        rows.append(row)
        log("  dNDVI %-6.3f  recall %5.1f%%  precision %5.1f%%  "
            "empirical FDR %5.1f%%  (mean %.1f cells flagged)"
            % (m, 100 * row["recall"] if row["recall"] == row["recall"] else 0,
               100 * row["precision"] if row["precision"] == row["precision"] else 0,
               100 * row["fdr"], row["detected"]))
    return rows


def min_detectable(rows, power=0.80):
    """Smallest injected change reaching `power` recall (linear interpolation)."""
    ok = [r for r in rows if r["magnitude"] > 0]
    for a, b in zip(ok, ok[1:]):
        if a["recall"] < power <= b["recall"]:
            f = (power - a["recall"]) / max(b["recall"] - a["recall"], 1e-9)
            return a["magnitude"] + f * (b["magnitude"] - a["magnitude"])
    return None
