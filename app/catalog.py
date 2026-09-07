"""STAC discovery. Searches the live archives per user query - nothing is
pre-staged on disk."""
from __future__ import annotations
import functools, datetime as dt
from typing import List, Dict, Any

import pystac_client
from .config import SOURCES, MAX_SCENES, CLOUD_LIMIT


@functools.lru_cache(maxsize=8)
def _client(api: str, sign: bool):
    mods = {}
    if sign:
        import planetary_computer
        mods["modifier"] = planetary_computer.sign_inplace
    return pystac_client.Client.open(api, **mods)


def _band_scale(asset):
    rb = getattr(asset, "extra_fields", {}) or {}
    bands = rb.get("raster:bands") or rb.get("bands")
    if isinstance(bands, list) and bands:
        b0 = bands[0]
        if "scale" in b0 or "offset" in b0:
            return (float(b0.get("scale", 1.0)), float(b0.get("offset", 0.0)))
    return None


def search(source: str, bbox, start: str, end: str,
           cloud: int = CLOUD_LIMIT, limit: int = MAX_SCENES) -> List[Dict[str, Any]]:
    """Return scene records (newest last) for one sensor over bbox/time."""
    cfg = SOURCES[source]
    q: Dict[str, Any] = {}
    if not cfg.get("sar"):
        q["eo:cloud_cover"] = {"lt": cloud}
    if source == "landsat-c2-l2":
        q["platform"] = {"in": ["landsat-8", "landsat-9"]}

    cl = _client(cfg["api"], cfg["sign"])
    # Enumerate the whole window (metadata only, cheap) so the stack spans the
    # full period instead of just the most recent page.
    res = cl.search(collections=[cfg["collection"]], bbox=bbox,
                    datetime=f"{start}/{end}", query=q or None, max_items=600)

    items = []
    for it in res.items():
        p = it.properties
        items.append({
            "id": it.id,
            "source": source,
            "collection": cfg["collection"],
            "datetime": (p.get("datetime") or p.get("start_datetime") or "")[:19],
            "cloud": round(float(p.get("eo:cloud_cover", 0.0)), 1),
            "platform": p.get("platform", ""),
            "epsg": p.get("proj:epsg") or p.get("proj:code"),
            "assets": {k: (a.href if hasattr(a, "href") else a["href"])
                       for k, a in it.assets.items()},
            # Per-asset scaling when the API publishes it; overrides the
            # collection default so a baseline change cannot silently corrupt
            # every index downstream.
            "scales": {k: _band_scale(a) for k, a in it.assets.items()
                       if _band_scale(a)},
            # keep the STAC self link for provenance
            "stac": next((l.href for l in it.links if l.rel == "self"), it.id),
        })

    # One scene per acquisition date: keep the least cloudy tile.
    best = {}
    for r in items:
        d = r["datetime"][:10]
        if d not in best or r["cloud"] < best[d]["cloud"]:
            best[d] = r
    items = sorted(best.values(), key=lambda r: r["datetime"])

    if len(items) > limit:                      # thin evenly across the window
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return items


def coverage(items) -> Dict[str, Any]:
    if not items:
        return {"n": 0}
    ds = [i["datetime"][:10] for i in items]
    return {"n": len(items), "first": ds[0], "last": ds[-1],
            "mean_cloud": round(sum(i["cloud"] for i in items) / len(items), 1)}
