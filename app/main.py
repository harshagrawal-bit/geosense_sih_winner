"""GeoSense API.

The architecture calls for Celery + Redis. Redis is not installed here and RAM
is tight, so jobs run on a bounded in-process thread pool with the same
submit/poll contract; swapping in Celery means replacing `submit` and `status`.
"""
from __future__ import annotations
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .services import AnalysisService
from .schemas import RunRequest
from .config import CACHE_DIR, SOURCES, BHUVAN_WMS

HERE = os.path.dirname(__file__)
app = FastAPI(title="GeoSense", version="0.1")

analysis_service = AnalysisService()


@app.post("/api/run")
def run(req: RunRequest):
    w, s, e, n = req.bbox
    if not (e > w and n > s):
        raise HTTPException(400, "bbox must be [west, south, east, north]")
    span = max(e - w, n - s)
    if span > 0.45:
        raise HTTPException(400, f"AOI too large ({span:.2f} deg). Keep it under "
                                 "0.45 deg so cells stay meaningful and reads stay fast.")
    if span < 0.01:
        raise HTTPException(400, "AOI too small - draw at least ~1 km across.")
    return {"job": analysis_service.submit(req)}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = analysis_service.job_status(jid)
    if not j:
        raise HTTPException(404, "unknown job")
    return JSONResponse(j)


@app.get("/api/semantic")
def semantic_status():
    from . import semantic
    return semantic.probe()


@app.get("/api/bhoonidhi")
def bhoonidhi_status():
    from . import bhoonidhi
    return bhoonidhi.status()


@app.get("/api/runs")
def runs():
    return analysis_service.recent_runs()


@app.get("/api/sources")
def sources():
    return {"sources": [{"key": k, "label": v["label"], "gsd": v.get("gsd"),
                         "sar": bool(v.get("sar"))} for k, v in SOURCES.items()],
            "bhuvan_wms": BHUVAN_WMS}


@app.get("/chips/{run_id}/{name}")
def chip(run_id: str, name: str):
    if "/" in run_id or "/" in name or ".." in run_id + name:
        raise HTTPException(400, "bad path")
    p = os.path.join(CACHE_DIR, run_id, name)
    if not os.path.exists(p):
        raise HTTPException(404, "no such chip")
    return FileResponse(p, media_type="image/png")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "app.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")),
          name="static")
