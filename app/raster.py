"""Windowed COG reads + spectral indices.

Only the pixels inside the AOI are fetched, at an overview level chosen by the
output shape, so a 20-date stack costs megabytes rather than gigabytes.
"""
from __future__ import annotations
import warnings, numpy as np, rasterio
from rasterio.warp import transform_bounds
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from .config import SOURCES, GRID_PX

warnings.filterwarnings("ignore", category=RuntimeWarning)

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="1",
    GDAL_HTTP_TIMEOUT="30",
    VSI_CACHE="TRUE",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
)


def read_window(href: str, bbox4326, out=GRID_PX) -> np.ndarray | None:
    """Read one asset over a lat/lon bbox into an out x out float32 array."""
    try:
        with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
            l, b, r, t = transform_bounds("EPSG:4326", src.crs, *bbox4326,
                                          densify_pts=21)
            win = from_bounds(l, b, r, t, transform=src.transform)
            if win.width < 1 or win.height < 1:
                return None
            arr = src.read(1, window=win, out_shape=(out, out),
                           resampling=Resampling.average,
                           boundless=True, fill_value=0).astype("float32")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            return arr
    except (RasterioIOError, ValueError, IndexError):
        return None


def _scale(a, cfg, band_scale=None):
    if a is None:
        return None
    sc, off = band_scale if band_scale else (cfg["scale"], cfg["offset"])
    out = a * sc + off
    out[a == 0] = np.nan                     # 0 is fill in both S2 L2A and L C2
    out[(out < -0.05) | (out > 1.6)] = np.nan
    return out


def _one_scene(it, bbox, want):
    """Read every band of a single scene and apply the cloud/shadow mask."""
    cfg = SOURCES[it["source"]]
    got = {}
    for b in want:
        key = cfg["bands"].get(b)
        href = it["assets"].get(key) if key else None
        a = read_window(href, bbox) if href else None
        if a is None:
            return None
        bs = it.get("scales", {}).get(key) if cfg.get("trust_stac_scale", True) else None
        got[b] = _scale(a, cfg, bs)

    qkey = cfg["bands"].get("scl")
    q = read_window(it["assets"][qkey], bbox) if qkey and qkey in it["assets"] else None
    bad = np.zeros((GRID_PX, GRID_PX), bool)
    if q is not None:
        if it["source"] == "sentinel-2-l2a":
            # SCL: 0 nodata, 1 saturated, 3 shadow, 8/9 cloud, 10 cirrus, 11 snow
            bad = np.isin(np.nan_to_num(q, nan=0).astype("uint8"), [0, 1, 3, 8, 9, 10, 11])
        else:
            qi = np.nan_to_num(q, nan=1).astype("uint16")     # Landsat QA_PIXEL
            bad = (((qi >> 1) & 1) | ((qi >> 3) & 1) | ((qi >> 4) & 1)).astype(bool)
    for b in want:
        got[b][bad] = np.nan

    valid = float(np.mean(np.isfinite(got[want[0]])))
    return {"date": it["date"] if "date" in it else it["datetime"][:10],
            "item": it, "bands": got, "valid": valid}


def optical_stack(items, bbox, want=("red", "nir", "green", "swir16", "blue"),
                  min_valid=0.0, workers=8, progress=None):
    """-> (dates, {band:(T,H,W)}, used_items).

    Scenes are fetched concurrently. Every scene that reads successfully is
    returned along with its cloud-free fraction so the caller can choose a
    coverage threshold without paying to download anything twice - which
    matters over monsoon India, where a strict threshold can empty the stack.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_scene, it, bbox, want): it for it in items}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception:
                r = None
            if progress:
                progress(n, len(futs))
            if r and r["valid"] >= min_valid:
                out.append(r)


    if not out:
        return [], {}, []
    out.sort(key=lambda r: r["date"])
    return out


def assemble(scenes, want, min_valid):
    """Pick the scenes clearing a coverage threshold and stack them."""
    keep = [r for r in scenes if r["valid"] >= min_valid]
    if not keep:
        return [], {}, []
    dates = [r["date"] for r in keep]
    stack = {b: np.stack([r["bands"][b] for r in keep]) for b in want}
    return dates, stack, [r["item"] for r in keep]


MIN_DENOM = 0.03      # reflectance sum below this is noise, not signal


def _norm(a, b):
    """Normalised difference with a denominator guard, clipped to [-1,1].

    Sentinel-2 baseline 04.00 applies a -1000 DN offset, so genuinely dark
    pixels land near zero reflectance. Without the guard those blow the ratio
    up to arbitrary values.
    """
    den = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / den
    out[~np.isfinite(out)] = np.nan
    out[np.abs(den) < MIN_DENOM] = np.nan
    return np.clip(out, -1.0, 1.0)


def indices(st):
    """Spectral indices, all in [-1,1]. NaN-safe."""
    red, nir, grn = st["red"], st["nir"], st["green"]
    sw1 = st.get("swir16"); blu = st.get("blue")
    out = {"ndvi": _norm(nir, red)}
    if sw1 is not None:
        out["mndwi"] = _norm(grn, sw1)          # water: green vs SWIR
        out["ndbi"] = _norm(sw1, nir)           # built-up / bare
        if blu is not None:
            out["bsi"] = _norm(sw1 + red, nir + blu)   # bare soil
    return out


def rgb_png(bands, path, gamma=1.5, pct=(2, 98), size=512, stretch=None,
            fixed=(0.0, 0.30)):
    """True-colour PNG from three 2-D reflectance arrays.

    Uses a *fixed* reflectance range (0 - 0.30, the standard Sentinel-2 true
    colour recipe) rather than a per-band percentile stretch. Percentile
    stretches are computed independently per band and per date, which both
    shifts the colour balance and cancels the very change we are trying to
    show. A fixed range keeps before and after radiometrically comparable and
    keeps colours natural. `stretch` overrides it when a caller needs an
    explicit range; it falls back to percentiles only for a scene that would
    otherwise be almost black.
    """
    from PIL import Image
    b = [np.nan_to_num(bands[c], nan=0.0) for c in ("red", "green", "blue")]
    img = np.dstack(b)
    out_stretch = []
    valid = img[img > 0]
    dark = valid.size and float(np.percentile(valid, 98)) < 0.06
    for c in range(3):
        ch = img[..., c]
        if stretch is not None:
            lo, hi = stretch[c]
        elif dark:                          # very dark scene: fall back
            v = ch[ch > 0]
            lo, hi = (np.percentile(v, pct) if v.size else fixed)
        else:
            lo, hi = fixed
        out_stretch.append((float(lo), float(hi)))
        img[..., c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    img = img ** (1 / gamma)
    Image.fromarray((img * 255).astype("uint8")).resize(
        (size, size), Image.LANCZOS).save(path, optimize=True)
    return out_stretch
