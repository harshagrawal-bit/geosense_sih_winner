"""Adapters that expose the prototype's existing implementations as ports."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import datetime as dt
import threading
import traceback
import uuid
from typing import Any, Callable
import numpy as np

from . import catalog, change, provenance, query, rank, raster
from .contracts import Progress
from .config import POSTGRES_DSN, STORAGE_BACKEND
from .repositories import DuckDBRunRepository, PostgreSQLRunRepository, RunRepository


class StacCatalogAdapter:
    search = staticmethod(catalog.search)
    coverage = staticmethod(catalog.coverage)


class RasterImageryAdapter:
    optical_stack = staticmethod(raster.optical_stack)
    assemble = staticmethod(raster.assemble)
    indices = staticmethod(raster.indices)
    read_window = staticmethod(raster.read_window)
    rgb_png = staticmethod(raster.rgb_png)


class RuleSemanticRetrievalAdapter:
    parse = staticmethod(query.parse)

    @staticmethod
    def retrieve(text: str):
        from . import semantic as clip_tier
        probe = clip_tier.probe()
        parsed = query.parse(text)
        concepts = list(parsed.get("matched_terms", []))
        if parsed["signature"] != "any":
            concepts.append(parsed["label"])
        evidence_types = ["spectral", "temporal"]
        if parsed.get("sar_expect") != "none":
            evidence_types.append("sar_optional")
        return {
            "query": text,
            "normalized_query": (text or "").lower().strip(),
            "concepts": concepts,
            "evidence_types": evidence_types,
            # Report what is actually on this machine. This used to be a
            # hardcoded False, which kept claiming "Model unavailable" long
            # after the CLIP tier existed.
            "backend": parsed.get("classifier", "keyword fallback"),
            "confident": parsed.get("confident", True),
            "margin": parsed.get("margin"),
            "alternatives": [a["label"] for a in parsed.get("alternatives", [])],
            "model_available": probe["installed"],
            "model_name": probe["model"],
            "model_state": probe["state"],
            "embedding": None,
            "intent": parsed,
        }


class HarmonicTemporalDetectionAdapter:
    to_cells = staticmethod(change.to_cells)
    select_dates = staticmethod(change.select_dates)
    fill_gaps = staticmethod(change.fill_gaps)
    deseasonalize = staticmethod(change.deseasonalize)
    multiscale_scan = staticmethod(change.multiscale_scan)
    multiscale_null = staticmethod(change.multiscale_null)
    conformal_p = staticmethod(change.conformal_p)
    amplitude_dispersion = staticmethod(change.amplitude_dispersion)
    db = staticmethod(change.db)
    spatial_smooth = staticmethod(change.spatial_smooth)
    scan_max_t = staticmethod(change.scan_max_t)


class RrfRankingAdapter:
    context_bonus = staticmethod(rank.context_bonus)
    linear_bonus = staticmethod(rank.linear_bonus)
    ranks_of = staticmethod(rank.ranks_of)
    rrf = staticmethod(rank.rrf)
    bh_threshold = staticmethod(rank.bh_threshold)


class Sentinel1RtcSarEvidenceAdapter:
    """Existing Sentinel-1 RTC/ADI evidence, exposed as an optional CUE port."""
    def __init__(self, catalog_adapter: Any, imagery_adapter: Any, temporal_adapter: Any):
        self.catalog = catalog_adapter
        self.imagery = imagery_adapter
        self.temporal = temporal_adapter

    def collect(self, bbox, start: str, end: str, split_date: str, ny: int, nx: int,
                progress: Progress) -> dict[str, Any] | None:
        items = self.catalog.search("sentinel-1-rtc", bbox, start, end, cloud=60, limit=14)
        if len(items) < 6:
            return None
        progress("prioritise", 74, f"SAR tier: {len(items)} Sentinel-1 RTC scenes")
        with ThreadPoolExecutor(max_workers=6) as executor:
            arrays = list(executor.map(
                lambda item: (item["datetime"][:10], self.imagery.read_window(item["assets"].get("vv"), bbox)),
                items,
            ))
        pairs = [(date, array) for date, array in arrays if array is not None]
        if len(pairs) < 6:
            return None
        dates = [date for date, _ in pairs]
        stack = np.stack([array for _, array in pairs])
        stack[stack <= 0] = np.nan
        pre = [index for index, date in enumerate(dates) if date < split_date]
        post = [index for index, date in enumerate(dates) if date >= split_date]
        if len(pre) < 3 or len(post) < 3:
            midpoint = len(dates) // 2
            pre, post = list(range(midpoint)), list(range(midpoint, len(dates)))
        adi_pre = self.temporal.to_cells(self.temporal.amplitude_dispersion(stack[pre])[None])[0]
        adi_post = self.temporal.to_cells(self.temporal.amplitude_dispersion(stack[post])[None])[0]
        db_cells = self.temporal.to_cells(self.temporal.db(stack))
        keep, _ = self.temporal.select_dates(db_cells, dates, 0.4, need=6)
        z_sar = None
        if len(keep) >= 6:
            residuals = self.temporal.deseasonalize(
                self.temporal.fill_gaps(db_cells[keep]), [dates[index] for index in keep], n_harm=1
            )
            residuals = self.temporal.spatial_smooth(residuals, ny, nx, r=1)
            z_sar, _, _ = self.temporal.scan_max_t(residuals)
            z_sar = z_sar.reshape(ny, nx)
        return {"n": len(dates), "dates": dates, "adi_pre": adi_pre, "adi_post": adi_post, "z": z_sar}


class HashChainProvenanceAdapter:
    new_chain = staticmethod(provenance.Chain)
    stac_refs = staticmethod(provenance.stac_refs)


class EvidenceConfirmationAdapter:
    """Confirm CUE candidates using independent evidence already in each candidate."""

    def confirm(self, detections: list[dict[str, Any]], progress: Progress) -> dict[str, Any]:
        progress("confirm", 96, f"Evaluating {len(detections)} CUE candidates")
        evaluations = []
        confirmed_candidates = []
        for detection in detections:
            result = self._evaluate(detection)
            evaluations.append(result)
            if result["status"] == "confirmed":
                confirmed_candidates.append(result)
        counts = {status: sum(e["status"] == status for e in evaluations)
                  for status in ("confirmed", "insufficient_evidence", "unavailable_evidence")}
        return {
            "status": "completed", "shortlisted": len(detections),
            "evaluated": len(evaluations), "confirmed": counts["confirmed"],
            "insufficient": counts["insufficient_evidence"],
            "unavailable": counts["unavailable_evidence"],
            "confirmed_candidates": confirmed_candidates, "evaluations": evaluations,
            "evidence": {"available": True, "methods": ["temporal", "spectral", "sar_optional"],
                         "rule": "certified temporal signal plus two directional spectral indicators"},
            "message": "Candidates evaluated with temporal and spectral corroboration; SAR is optional supporting evidence.",
        }

    @staticmethod
    def _evaluate(detection: dict[str, Any]) -> dict[str, Any]:
        temporal = detection.get("confirmation", {}).get("temporal", {})
        deltas = detection.get("deltas", {})
        targets = detection.get("confirmation", {}).get("targets", {})
        sar = detection.get("sar")
        def finite(value):
            try:
                return np.isfinite(float(value))
            except (TypeError, ValueError):
                return False

        if not all(finite(temporal.get(key)) for key in ("z", "p", "magnitude")):
            return {"id": detection["id"], "status": "unavailable_evidence",
                    "reasons": ["temporal evidence is unavailable"], "evidence": {}}
        directional = []
        for index, target in targets.items():
            delta = deltas.get(index)
            if target and finite(delta):
                directional.append({"index": index, "delta": float(delta), "expected": float(target),
                                    "aligned": bool(float(delta) * float(target) > 0)})
        if not directional and not any(targets.values()):
            directional = [{"index": index, "delta": float(delta), "expected": 0.0,
                            "aligned": bool(abs(float(delta)) > 0)}
                           for index, delta in deltas.items() if finite(delta)]
        aligned = [item for item in directional if item["aligned"]]
        evidence = {"temporal": temporal, "spectral": directional,
                    "sar": sar if sar is not None else None}
        if not detection.get("certified"):
            return {"id": detection["id"], "status": "insufficient_evidence",
                    "reasons": ["temporal evidence did not pass the CUE certification threshold"],
                    "evidence": evidence}
        if len(directional) == 0:
            return {"id": detection["id"], "status": "unavailable_evidence",
                "reasons": ["directional spectral evidence is unavailable"],
                "evidence": evidence}
        if len(directional) < 2:
            return {"id": detection["id"], "status": "insufficient_evidence",
                    "reasons": ["fewer than two directional spectral indicators are available"],
                    "evidence": evidence}
        if len(aligned) < 2:
            return {"id": detection["id"], "status": "insufficient_evidence",
                    "reasons": ["fewer than two spectral indicators support the query direction"],
                    "evidence": evidence}
        return {"id": detection["id"], "status": "confirmed",
                "reasons": ["certified temporal signal corroborated by two directional spectral indicators"],
                "evidence": evidence}


class DuckDbStorageAdapter:
    def __init__(self, repository: RunRepository | None = None):
        self.repository = repository or DuckDBRunRepository()

    def save(self, result):
        self.repository.save(result)

    def recent(self, limit=25):
        return self.repository.recent(limit)

    def vectors(self, limit=5000):
        return self.repository.vectors(limit)

    def search_vectors(self, query_vec, limit=10, exclude=None):
        return self.repository.search_vectors(query_vec, limit, exclude)

    def vector_by_id(self, det_id):
        return self.repository.vector_by_id(det_id)

    def stats(self):
        return self.repository.stats()


class PostgreSQLStorageAdapter:
    def __init__(self, repository: RunRepository | None = None):
        self.repository = repository or PostgreSQLRunRepository(POSTGRES_DSN)

    def save(self, result):
        self.repository.save(result)

    def recent(self, limit=25):
        return self.repository.recent(limit)

    def vectors(self, limit=5000):
        return self.repository.vectors(limit)

    def search_vectors(self, query_vec, limit=10, exclude=None):
        return self.repository.search_vectors(query_vec, limit, exclude)

    def vector_by_id(self, det_id):
        return self.repository.vector_by_id(det_id)

    def stats(self):
        return self.repository.stats()


def configured_storage_adapter():
    if STORAGE_BACKEND == "duckdb":
        return DuckDbStorageAdapter()
    if STORAGE_BACKEND in {"postgres", "postgresql", "postgis"}:
        return PostgreSQLStorageAdapter()
    raise ValueError(f"Unsupported GEOSENSE_STORAGE backend: {STORAGE_BACKEND}")


class ThreadPoolJobAdapter:
    """Current local job implementation behind the future queue boundary."""
    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, request: dict[str, Any], work: Callable[[str, Progress], dict[str, Any]]) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "state": "running", "pct": 0,
                                  "stage": "queued", "message": "Queued",
                                  "started": dt.datetime.now().isoformat(), "request": request}

        def progress(stage: str, pct: int, message: str) -> None:
            with self._lock:
                self._jobs[job_id].update(stage=stage, pct=pct, message=message)

        def runner() -> None:
            try:
                result = work(job_id, progress)
                progress("done", 100, "Complete" if result.get("ok") else result.get("error", "Failed"))
                with self._lock:
                    self._jobs[job_id].update(state="done" if result.get("ok") else "error", result=result)
            except Exception:
                with self._lock:
                    self._jobs[job_id].update(state="error", pct=100, stage="error",
                                              message="Pipeline failure", traceback=traceback.format_exc())

        self._pool.submit(runner)
        return job_id

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None
