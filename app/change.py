"""Change detection.

`harmonic_break` is a COLD-style continuous monitor: a harmonic model of the
seasonal cycle is fitted on a stable training prefix, then later observations
are scored against the model's own residual spread. A break is a run of
observations the seasonal model cannot explain - which is what separates real
structural change from ordinary phenology.

`amplitude_dispersion` is the SAR tier (ADI = sigma/mu of backscatter
amplitude); low dispersion means a persistent, stable scatterer.
"""
from __future__ import annotations
import datetime as dt
import numpy as np

from .config import CELL, GRID_PX


# ---------------------------------------------------------------- gridding --
def to_cells(arr, cell=CELL):
    """(T,H,W) -> (T,nY,nX) block means, ignoring NaN."""
    T, H, W = arr.shape
    ny, nx = H // cell, W // cell
    a = arr[:, :ny * cell, :nx * cell].reshape(T, ny, cell, nx, cell)
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=(2, 4))


def _doy(dates):
    d0 = dt.date.fromisoformat(dates[0])
    return np.array([(dt.date.fromisoformat(d) - d0).days for d in dates], float)


def _design(t, n_harm=1, trend=False):
    cols = [np.ones_like(t)]
    if trend:
        cols.append(t / 365.25)
    for k in range(1, n_harm + 1):
        w = 2 * np.pi * k * t / 365.25
        cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols)


# ------------------------------------------------------------------- COLD ---
def harmonic_break(series, dates, min_seg=3):
    """One cell's time series -> break metrics, or None.

    Step 1: fit a harmonic model of the annual cycle over the whole series and
    subtract it, leaving residuals with the phenology removed. No linear trend
    term - over a 1-2 year window it is not identifiable and extrapolates
    catastrophically.

    Step 2: scan every split of the residual series and keep the one with the
    largest two-sample t statistic. A structural change is a persistent shift
    the seasonal model cannot account for; ordinary phenology cancels out.
    """
    y = np.asarray(series, float)
    t = _doy(dates)
    ok = np.isfinite(y)
    n = int(ok.sum())
    if n < 2 * min_seg + 2:
        return None
    y, t = y[ok], t[ok]
    ds = [d for d, m in zip(dates, ok) if m]

    n_harm = 2 if n >= 12 else 1
    X = _design(t, n_harm)
    if n <= X.shape[1] + 1:
        X = _design(t, 1)
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    fit = X @ coef
    r = y - fit
    seas = float(np.std(fit))

    best = None
    for b in range(min_seg, n - min_seg + 1):
        pre, post = r[:b], r[b:]
        sp = np.sqrt(((b - 1) * np.var(pre, ddof=1) +
                      (n - b - 1) * np.var(post, ddof=1)) / max(n - 2, 1))
        sp = max(float(sp), 0.02)               # sensor + atmospheric noise floor
        se = sp * np.sqrt(1.0 / b + 1.0 / (n - b))
        z = float((post.mean() - pre.mean()) / se)
        if best is None or abs(z) > abs(best["z"]):
            best = {"z": z, "date": ds[b], "idx": b,
                    "magnitude": float(post.mean() - pre.mean()),
                    "pre": float(np.mean(y[:b])), "post": float(np.mean(y[b:])),
                    "rmse": sp, "seasonal_amp": seas, "n": n}
    return best


def break_field(cube, dates):
    """(T,nY,nX) -> per-cell dicts of break metrics (None where unfittable)."""
    T, ny, nx = cube.shape
    out = np.empty((ny, nx), dtype=object)
    for j in range(ny):
        for i in range(nx):
            out[j, i] = harmonic_break(cube[:, j, i], dates)
    return out


def field_array(field, key, default=np.nan):
    ny, nx = field.shape
    a = np.full((ny, nx), default, float)
    for j in range(ny):
        for i in range(nx):
            r = field[j, i]
            if r is not None:
                a[j, i] = r[key]
    return a


# -------------------------------------------------------------------- SAR ---
def amplitude_dispersion(stack):
    """(T,H,W) linear-power SAR -> ADI = sigma/mu of amplitude, per pixel."""
    amp = np.sqrt(np.clip(stack, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(amp, axis=0)
        sd = np.nanstd(amp, axis=0)
        adi = sd / mu
    adi[~np.isfinite(adi)] = np.nan
    return adi


def db(stack):
    return 10.0 * np.log10(np.clip(stack, 1e-6, None))


# ------------------------------------------------ vectorised grid pipeline ---
def select_dates(cube, dates, min_cell_valid=0.55, need=8):
    """Indices of dates where enough cells survived cloud masking.

    Relaxes the coverage requirement until `need` dates qualify, so a
    cloud-dominated AOI degrades gracefully instead of returning nothing. The
    threshold actually used is reported, because it changes how much the
    reader should trust the stack.
    """
    cov = np.array([np.mean(np.isfinite(cube[t])) for t in range(cube.shape[0])])
    for thr in (min_cell_valid, 0.40, 0.30, 0.20, 0.10):
        keep = np.where(cov >= thr)[0]
        if len(keep) >= need:
            return keep, float(thr)
    return np.where(cov >= 0.10)[0], 0.10


def fill_gaps(cube):
    """Linear interpolation in time per cell, so every cell shares one design
    matrix. Cells with almost no valid observations are set to their mean."""
    n = cube.shape[0]
    flat = cube.reshape(n, -1).copy()
    x = np.arange(n, dtype=float)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        m = np.isfinite(col)
        if m.sum() < 4:
            flat[:, j] = np.nanmean(col) if m.any() else 0.0
        elif not m.all():
            flat[:, j] = np.interp(x, x[m], col[m])
    return flat.reshape(cube.shape)


def deseasonalize(cube, dates, n_harm=None):
    """(T,nY,nX) -> residual matrix (T, N) with the annual cycle removed."""
    t = _doy(dates)
    T = len(dates)
    if n_harm is None:
        n_harm = 2 if T >= 14 else 1
    X = _design(t, n_harm)
    Y = cube.reshape(T, -1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ coef


def scan_max_t(R, min_seg=3, sign=0):
    """Vectorised max-t changepoint scan.

    R is (T, N). Returns (z, split index, magnitude) per column for the split
    that maximises the two-sample t statistic. `sign` selects the alternative:
    +1 keeps the largest positive t, -1 the most negative, 0 the largest in
    absolute value. A query that names a direction ("construction") must be
    tested one-sided, or a change of the opposite kind certifies against it.
    """
    T, N = R.shape
    best_z = np.full(N, -np.inf if sign > 0 else (np.inf if sign < 0 else 0.0))
    best_b = np.zeros(N, int)
    best_m = np.zeros(N)
    csum = np.cumsum(R, axis=0)
    csq = np.cumsum(R ** 2, axis=0)
    tot, totsq = csum[-1], csq[-1]

    for b in range(min_seg, T - min_seg + 1):
        n1, n2 = b, T - b
        s1, s2 = csum[b - 1], tot - csum[b - 1]
        q1, q2 = csq[b - 1], totsq - csq[b - 1]
        m1, m2 = s1 / n1, s2 / n2
        v1 = np.maximum(q1 - n1 * m1 ** 2, 0) / max(n1 - 1, 1)
        v2 = np.maximum(q2 - n2 * m2 ** 2, 0) / max(n2 - 1, 1)
        sp = np.sqrt(np.maximum(((n1 - 1) * v1 + (n2 - 1) * v2) / max(T - 2, 1), 0))
        sp = np.maximum(sp, 0.02)               # noise floor
        z = (m2 - m1) / (sp * np.sqrt(1.0 / n1 + 1.0 / n2))
        if sign > 0:
            upd = z > best_z
        elif sign < 0:
            upd = z < best_z
        else:
            upd = np.abs(z) > np.abs(best_z)
        best_z = np.where(upd, z, best_z)
        best_b = np.where(upd, b, best_b)
        best_m = np.where(upd, m2 - m1, best_m)
    best_z[~np.isfinite(best_z)] = 0.0
    return best_z, best_b, best_m


def permutation_null(R, min_seg=3, n_perm=200, seed=0, sample=None, sign=0):
    """Null distribution of the max-t statistic by shuffling time order.

    Breaking the temporal ordering destroys any real step while preserving the
    marginal residual distribution, so the resulting statistics are draws from
    the no-change null for this scene, this cadence, this noise level.
    """
    rng = np.random.default_rng(seed)
    T, N = R.shape
    cols = np.arange(N) if sample is None or sample >= N else \
        rng.choice(N, size=sample, replace=False)
    sub = R[:, cols]
    out = []
    for _ in range(n_perm):
        z, _, _ = scan_max_t(sub[rng.permutation(T)], min_seg, sign=sign)
        out.append(z if sign > 0 else (-z if sign < 0 else np.abs(z)))
    return np.concatenate(out)


def conformal_p(obs, null, sign=0):
    """p = (1 + #{null >= obs}) / (n_null + 1). Valid under exchangeability.

    `sign` must match the one used for the scan so the observed and null
    statistics are on the same scale.
    """
    stat = obs if sign > 0 else (-obs if sign < 0 else np.abs(obs))
    null = np.sort(np.asarray(null))
    ge = len(null) - np.searchsorted(null, stat, side="left")
    return (1.0 + ge) / (len(null) + 1.0)


def spatial_smooth(R, ny, nx, r=1):
    """Box-filter the residual field spatially (matched filter for extended
    change). Real change covers neighbouring cells; per-cell sensor noise does
    not, so this lifts SNR for anything larger than one cell. The permutation
    null is computed on the smoothed field too, so the calibration stays exact.
    """
    T = R.shape[0]
    F = R.reshape(T, ny, nx)
    acc = np.zeros_like(F)
    cnt = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            acc += np.roll(np.roll(F, dy, 1), dx, 2)
            cnt += 1
    return (acc / cnt).reshape(T, ny * nx)


def multiscale_scan(comp, ny, nx, sign=0, radii=(0, 1)):
    """Scan at several spatial scales and keep the best per cell.

    A 3x3 filter is a matched filter for change that spans several cells, but
    it divides a single-cell change by ~9 while only cutting noise by ~3 - so
    smoothing alone hides exactly the small, sharp events (a new building plot)
    that matter most. Scanning unsmoothed and smoothed and taking the stronger
    response covers both regimes; `multiscale_null` takes the same maximum on
    every permutation, so testing two scales costs calibration, not validity.
    """
    fields = [comp if r == 0 else spatial_smooth(comp, ny, nx, r) for r in radii]
    zs, bs, ms = zip(*[scan_max_t(f, sign=sign) for f in fields])
    z, b, m = zs[0].copy(), bs[0].copy(), ms[0].copy()
    for k in range(1, len(fields)):
        better = (zs[k] > z) if sign > 0 else \
                 ((zs[k] < z) if sign < 0 else (np.abs(zs[k]) > np.abs(z)))
        z = np.where(better, zs[k], z)
        b = np.where(better, bs[k], b)
        m = np.where(better, ms[k], m)
    return z, b, m


def multiscale_null(comp, ny, nx, sign=0, radii=(0, 1), n_perm=160, seed=0,
                    sample=160):
    """Null for `multiscale_scan`: the same max-over-scales on shuffled time."""
    rng = np.random.default_rng(seed)
    T, N = comp.shape
    out = []
    for _ in range(n_perm):
        z, _, _ = multiscale_scan(comp[rng.permutation(T)], ny, nx,
                                  sign=sign, radii=radii)
        stat = z if sign > 0 else (-z if sign < 0 else np.abs(z))
        if sample and sample < N:
            stat = stat[rng.choice(N, size=sample, replace=False)]
        out.append(stat)
    return np.concatenate(out)
