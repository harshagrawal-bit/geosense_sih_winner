"""Per-pixel change segmentation and area measurement.

A detection that says "something changed in this 550 m cell" is not actionable;
an analyst needs to know *how much* changed and *where inside* the cell. The
cell grid exists because the statistics need a time series per unit, not
because 550 m is the resolution of the answer - the underlying imagery is
still ~30 m per pixel, so the extent can be recovered at full resolution once
the cell has been flagged.

We deliberately do not use SAM here. SAM segments *objects* by appearance; what
we need is the extent of a *change*, which is defined by the same composite
index difference the detector already tested. Thresholding that difference
segments exactly the thing that was detected, needs no model, and cannot
disagree with the statistic that raised the alarm.
"""
from __future__ import annotations

import math
import numpy as np


def composite_delta(pre: dict, post: dict, weights: dict) -> np.ndarray:
    """Per-pixel change in the direction the query expects (H, W)."""
    keys = [k for k in weights if k in pre and k in post]
    if not keys:
        return None
    total = sum(abs(weights[k]) for k in keys) or 1.0
    out = np.zeros_like(next(iter(pre.values())), dtype="float32")
    for k in keys:
        out += (weights[k] / total) * (post[k] - pre[k])
    return out


def threshold(delta: np.ndarray, k: float = 2.5) -> np.ndarray:
    """Robust threshold: keep pixels the local noise cannot explain.

    Uses the median absolute deviation rather than the standard deviation, so
    the change itself does not inflate the very threshold meant to detect it.
    """
    d = np.asarray(delta, dtype="float32")
    finite = d[np.isfinite(d)]
    if finite.size < 16:
        return np.zeros(d.shape, bool)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * 1.4826     # -> sigma
    mad = max(mad, 0.01)
    return np.nan_to_num(d, nan=0.0) > (med + k * mad)


def largest_blob(mask: np.ndarray, min_px: int = 6) -> np.ndarray:
    """Keep the biggest 4-connected component; drop speckle.

    Iterative flood fill - the masks are ~100x100, and pulling in SciPy for one
    labelling call is not worth a dependency.
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return m
    seen = np.zeros_like(m)
    best: list[tuple[int, int]] = []
    h, w = m.shape
    for sy in range(h):
        for sx in range(w):
            if not m[sy, sx] or seen[sy, sx]:
                continue
            stack, comp = [(sy, sx)], []
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    out = np.zeros_like(m)
    if len(best) >= min_px:
        ys, xs = zip(*best)
        out[list(ys), list(xs)] = True
    return out


def pixel_area_m2(bbox, grid_px: int) -> float:
    """Ground area of one analysis pixel, accounting for latitude."""
    w, s, e, n = bbox
    lat = math.radians((s + n) / 2.0)
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(lat)
    return ((e - w) / grid_px * m_per_deg_lon) * ((n - s) / grid_px * m_per_deg_lat)


def measure(mask: np.ndarray, bbox, grid_px: int) -> dict:
    px = pixel_area_m2(bbox, grid_px)
    n = int(np.count_nonzero(mask))
    return {"pixels": n, "area_m2": round(n * px, 1),
            "area_ha": round(n * px / 10_000.0, 3),
            "pixel_m": round(math.sqrt(px), 1)}


def mask_bbox(mask: np.ndarray, window_bbox) -> list | None:
    """Geographic bounds of the changed pixels, tighter than the cell."""
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    lw, ls, le, ln = window_bbox
    return [lw + (le - lw) * xs.min() / w, ln - (ln - ls) * (ys.max() + 1) / h,
            lw + (le - lw) * (xs.max() + 1) / w, ln - (ln - ls) * ys.min() / h]


def outline_png(rgb: np.ndarray, mask: np.ndarray, path: str, size: int = 320):
    """The after-chip with the changed extent outlined."""
    from PIL import Image
    a = np.nan_to_num(np.asarray(rgb, dtype="float32"), nan=0.0)
    a = np.clip(a / 0.30, 0, 1) ** (1 / 1.5)
    img = (a * 255).astype("uint8")

    edge = mask & ~(
        np.roll(mask, 1, 0) & np.roll(mask, -1, 0) &
        np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    img[mask] = (0.72 * img[mask] + 0.28 * np.array([255, 90, 70])).astype("uint8")
    img[edge] = np.array([255, 120, 96], dtype="uint8")
    Image.fromarray(img).resize((size, size), Image.NEAREST).save(path,
                                                                 optimize=True)
    return path
