"""TIP -> CUE -> CONFIRM orchestration over the real prototype pipeline."""
from __future__ import annotations
import os, time, datetime as dt
from dataclasses import dataclass
import numpy as np

from . import catalog, raster, change, query as Q, rank, provenance
from .config import SOURCES, CELL, GRID_PX, CACHE_DIR
from .contracts import (CatalogPort, ConfirmationPort, ImageryPort, ProvenancePort,
                        RankingPort, SarEvidencePort, SemanticRetrievalPort,
                        TemporalDetectionPort)
from .schemas import (CONFIRMResult, CUEResult, RunRequest, SemanticRetrievalResult,
                      TIPResult)


@dataclass(frozen=True)
class PipelineDependencies:
    """Ports used by the orchestrator, populated with Phase 1 adapters."""
    catalog: CatalogPort
    imagery: ImageryPort
    semantic: SemanticRetrievalPort
    temporal: TemporalDetectionPort
    sar: SarEvidencePort
    ranking: RankingPort
    provenance: ProvenancePort
    confirmation: ConfirmationPort


def default_dependencies() -> PipelineDependencies:
    # Imports live here to avoid a circular import while adapters remain small.
    from .adapters import (HarmonicTemporalDetectionAdapter, HashChainProvenanceAdapter,
                           EvidenceConfirmationAdapter, RasterImageryAdapter, RrfRankingAdapter,
                           RuleSemanticRetrievalAdapter, Sentinel1RtcSarEvidenceAdapter,
                           StacCatalogAdapter)
    catalog_adapter = StacCatalogAdapter()
    imagery_adapter = RasterImageryAdapter()
    temporal_adapter = HarmonicTemporalDetectionAdapter()
    return PipelineDependencies(
        catalog=catalog_adapter, imagery=imagery_adapter,
        semantic=RuleSemanticRetrievalAdapter(), temporal=temporal_adapter,
        sar=Sentinel1RtcSarEvidenceAdapter(catalog_adapter, imagery_adapter, temporal_adapter),
        ranking=RrfRankingAdapter(), provenance=HashChainProvenanceAdapter(),
        confirmation=EvidenceConfirmationAdapter(),
    )


def _win(months, end=None):
    end = dt.date.fromisoformat(end) if end else dt.date.today()
    start = end - dt.timedelta(days=int((months or 24) * 30.44))
    return start.isoformat(), end.isoformat()


def _cell_bbox(bbox, j, i, ny, nx):
    w, s, e, n = bbox
    return [w + (e - w) * i / nx, n - (n - s) * (j + 1) / ny,
            w + (e - w) * (i + 1) / nx, n - (n - s) * j / ny]


def sar_tier(bbox, start, end, split_date, ny, nx, say, max_scenes=14):
    """Compatibility wrapper; the real implementation now lives in its SAR adapter."""
    return default_dependencies().sar.collect(bbox, start, end, split_date, ny, nx, say)


def run(bbox, text, months=None, alpha=0.10, sources=("sentinel-2-l2a",),
        use_sar=False, max_scenes=24, cloud=40, job=None, run_id=None,
        dependencies: PipelineDependencies | None = None):
    """Run the current algorithms through the explicit TIP → CUE → CONFIRM flow."""
    t0 = time.time()
    say = job or (lambda *a, **k: None)
    dependencies = dependencies or default_dependencies()
    chain = dependencies.provenance.new_chain()
    run_id = run_id or dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    # ============================================================= TIP ===
    # Resolve intent and cheaply screen archive metadata / AOI COG windows.
    semantic = SemanticRetrievalResult.model_validate(dependencies.semantic.retrieve(text))
    q = semantic.intent
    months = q["months"] or months or 24
    start, end = _win(months)
    chain.add("query", {"text": text, "signature": q["signature"],
                        "targets": q["targets"], "window": [start, end],
                        "alpha": alpha, "backend": semantic.backend,
                        "concepts": semantic.concepts})
    say("discover", 3, f"Query resolved to '{q['label']}'")

    # ------------------------------------------------------- 1. discover ---
    items, per_source = [], {}
    for src in sources:
        say("discover", 6, f"Searching {SOURCES[src]['label']}…")
        try:
            got = dependencies.catalog.search(src, bbox, start, end, cloud=cloud,
                                              limit=max_scenes)
        except Exception as ex:
            per_source[src] = {"error": str(ex)[:200], "n": 0}
            continue
        per_source[src] = dependencies.catalog.coverage(got)
        items += got
    items.sort(key=lambda r: r["datetime"])
    if len(items) < 8:
        return {"ok": False, "error":
                f"Only {len(items)} usable scenes in the archive for this AOI and "
                f"window. Widen the dates, raise the cloud limit, or enlarge the AOI.",
                "sources": per_source, "query": q}
    chain.add("discover", {"candidates": len(items), "sources": per_source})
    say("discover", 18, f"{len(items)} candidate scenes")
    tip = TIPResult(
        candidates=len(items), screened=len(items), sources=per_source,
        evidence={"archive": "STAC", "screening": "scene metadata and AOI availability"},
    ).model_dump()

    # -------------------------------------------- 2. fetch pixels (COGs) ---
    def prog(n, tot):
        say("discover", 18 + int(42 * n / max(tot, 1)),
            f"Reading scene {n}/{tot} (windowed COG)")

    WANT = ("red", "nir", "green", "swir16", "blue")
    scenes = dependencies.imagery.optical_stack(items, bbox, want=WANT, progress=prog)

    # Relax the cloud-free coverage requirement rather than fail outright: over
    # monsoon India a strict threshold empties the stack, and a partly-clouded
    # scene still contributes valid pixels because masking is per-pixel.
    for min_valid in (0.35, 0.20, 0.10):
        dates, st, used = dependencies.imagery.assemble(scenes, WANT, min_valid)
        if len(dates) >= 8:
            break
    if len(dates) < 8:
        best = max((r["valid"] for r in scenes), default=0.0)
        return {"ok": False, "error":
                f"Only {len(dates)} of {len(items)} scenes cleared 10% cloud-free "
                f"coverage over this AOI (best scene: {best:.0%}). This AOI is "
                f"cloud-dominated for the window requested - widen the dates, "
                f"raise the cloud limit, or use the SAR tier.",
                "sources": per_source, "query": q}
    say("prioritise", 62, f"{len(dates)} usable dates (coverage ≥ {min_valid:.0%})")

    ix = dependencies.imagery.indices(st)
    chain.add("ingest", {"scenes_used": len(dates), "min_coverage": min_valid,
                         "stac": dependencies.provenance.stac_refs(used)[:60],
                         "grid_px": GRID_PX, "indices": sorted(ix)})

    # ============================================================= CUE ===
    # Current temporal, spectral, optional SAR and ranking evidence stages.
    cells = {k: dependencies.temporal.to_cells(v) for k, v in ix.items()}
    ny, nx = next(iter(cells.values())).shape[1:]

    # Choose the usable dates once, from NDVI (the densest index), then apply
    # the same temporal grid to every index so they stay directly comparable.
    keep_idx, cov_thr = dependencies.temporal.select_dates(cells["ndvi"], dates)
    if len(keep_idx) < 8:
        return {"ok": False, "error":
                f"Only {len(keep_idx)} dates retain usable coverage after cloud "
                f"masking (need 8). This AOI is cloud-dominated for this window.",
                "sources": per_source, "query": q}
    kept_dates = [dates[i] for i in keep_idx]
    resid = {k: dependencies.temporal.deseasonalize(dependencies.temporal.fill_gaps(c[keep_idx]), kept_dates)
             for k, c in cells.items()}
    say("prioritise", 66, f"{len(kept_dates)} dates on the model grid "
                          f"(cell coverage ≥ {cov_thr:.0%})")

    half = max(len(keep_idx) // 2, 1)
    pre_means = {k: np.nanmean(cells[k][keep_idx[:half]], axis=0) for k in cells}

    # composite residual in the direction the query expects
    w = {k: v for k, v in q["targets"].items() if k in resid}
    if not w or all(abs(v) < 1e-9 for v in w.values()):
        comp = max(resid.values(), key=lambda r: np.abs(r).mean())
        mode = "any-change (largest-variance index)"
    else:
        tot = sum(abs(v) for v in w.values())
        comp = sum((v / tot) * resid[k] for k, v in w.items())
        mode = " + ".join(f"{v:+.1f}·{k.upper()}" for k, v in w.items())
    chain.add("model", {"dates": kept_dates, "composite": mode,
                        "cell_coverage_threshold": cov_thr,
                        "harmonics": 2 if len(kept_dates) >= 14 else 1,
                        "spatial_scales": "1x1 and 3x3 (max, null-calibrated)"})
    say("prioritise", 70, "Fitting harmonic model per cell")

    # -------------------------------------------- 4. detect + certify (BH) ---
    # The composite is built so a change matching the query comes out positive,
    # so a named signature is a one-sided test; "any change" stays two-sided.
    sgn = 0 if not w else 1
    z, b, mag = dependencies.temporal.multiscale_scan(comp, ny, nx, sign=sgn)
    null = dependencies.temporal.multiscale_null(comp, ny, nx, sign=sgn, n_perm=140, seed=7,
                                                  sample=160)
    p = dependencies.temporal.conformal_p(z, null, sign=sgn).reshape(ny, nx)
    thr = dependencies.ranking.bh_threshold(p, alpha)
    z2, b2, mag2 = z.reshape(ny, nx), b.reshape(ny, nx), mag.reshape(ny, nx)
    say("verify", 80, "Permutation calibration + BH control")

    # ----------------------------------------------------- 5. prioritise ---
    split_pre = kept_dates[int(np.median(b2[p <= thr]))] if (p <= thr).any() \
        else kept_dates[len(kept_dates) // 2]
    score = z2 if w else np.abs(z2)
    score = score + dependencies.ranking.context_bonus(q["context"], pre_means)
    if q.get("shape") == "linear":
        score = score + 1.2 * dependencies.ranking.linear_bonus(z2)

    # SAR evidence stream (optional, non-fatal)
    sar = None
    if use_sar:
        try:
            sar = dependencies.sar.collect(bbox, start, end, split_pre, ny, nx, say)
        except Exception as ex:
            say("prioritise", 76, f"SAR tier unavailable: {str(ex)[:60]}")
    streams = [dependencies.ranking.ranks_of(score), dependencies.ranking.ranks_of(np.abs(z2)), dependencies.ranking.ranks_of(-p)]
    if sar:
        d_adi = np.nan_to_num(sar["adi_pre"] - sar["adi_post"], nan=0.0)
        want = q["sar_expect"]
        if want == "stabilise":       # new hard target -> dispersion falls
            streams.append(dependencies.ranking.ranks_of(d_adi))
        elif want == "destabilise":   # surface disrupted -> dispersion rises
            streams.append(dependencies.ranking.ranks_of(-d_adi))
        if sar["z"] is not None:
            streams.append(dependencies.ranking.ranks_of(np.abs(sar["z"])))
        chain.add("sar", {"scenes": sar["n"], "collection": "sentinel-1-rtc",
                          "expect": want})
    fused = dependencies.ranking.rrf(*streams)
    chain.add("certify", {"alpha": alpha, "bh_threshold": thr, "sided": sgn,
                          "null_draws": int(null.size),
                          "null_p95": float(np.percentile(null, 95)),
                          "n_pass": int((p <= thr).sum())})

    # ------------------------------------------------------- 6. chips -----
    outdir = os.path.join(CACHE_DIR, run_id)
    os.makedirs(outdir, exist_ok=True)
    split_date = split_pre
    di = [i for i, d in enumerate(dates) if d in kept_dates]
    pre_i = [i for i in di if dates[i] < split_date] or di[:1]
    post_i = [i for i in di if dates[i] >= split_date] or di[-1:]

    def comp_img(idxs):
        return {c: np.nanmedian(st[c][idxs], axis=0) for c in ("red", "green", "blue")}

    pre_img, post_img = comp_img(pre_i), comp_img(post_i)
    stretch = dependencies.imagery.rgb_png(pre_img, os.path.join(outdir, "before.png"))
    dependencies.imagery.rgb_png(post_img, os.path.join(outdir, "after.png"), stretch=stretch)

    # -------------------------------------------------- 7. detections -----
    order = np.argsort(-fused.ravel())
    dets, seen = [], set()
    for flat in order:
        j, i = divmod(int(flat), nx)
        if len(dets) >= 12:
            break
        if any(abs(j - jj) <= 1 and abs(i - ii) <= 1 for jj, ii in seen):
            continue                                  # spatial non-max suppress
        pv = float(p[j, i])
        certified = pv <= thr
        if len(dets) >= 4 and not certified:
            break
        seen.add((j, i))
        y0, x0 = j * CELL, i * CELL
        pad = CELL // 2
        sl = (slice(max(y0 - pad, 0), min(y0 + CELL + pad, GRID_PX)),
              slice(max(x0 - pad, 0), min(x0 + CELL + pad, GRID_PX)))
        cid = f"{j:02d}{i:02d}"

        # Split this cell on *its own* detected break, not the scene-wide one:
        # a chip cut at someone else's break date shows nothing.
        bi = int(b2[j, i])
        pre_k = list(keep_idx[:bi]) or list(keep_idx[:1])
        post_k = list(keep_idx[bi:]) or list(keep_idx[-1:])

        def med(idxs, box):
            return {c: np.nanmedian(st[c][idxs][(slice(None),) + box], axis=0)
                    for c in ("red", "green", "blue")}

        sp = dependencies.imagery.rgb_png(med(pre_k, sl),
                            os.path.join(outdir, f"c{cid}_before.png"), size=320)
        dependencies.imagery.rgb_png(med(post_k, sl),
                       os.path.join(outdir, f"c{cid}_after.png"), size=320,
                       stretch=sp)

        deltas = {k: float(np.nanmean(cells[k][post_k, j, i]) -
                           np.nanmean(cells[k][pre_k, j, i])) for k in cells}
        dets.append({
            "id": f"{run_id}-{cid}", "cell": [j, i],
            "bbox": _cell_bbox(bbox, j, i, ny, nx),
            "z": float(z2[j, i]), "p": pv, "certified": bool(certified),
            "confidence": round(100 * (1 - pv), 2),
            "break_date": kept_dates[bi],
            "n_pre": len(pre_k), "n_post": len(post_k),
            "magnitude": float(mag2[j, i]),
            "deltas": deltas,
            "confirmation": {"targets": q["targets"], "temporal": {
                "z": float(z2[j, i]), "p": pv, "magnitude": float(mag2[j, i]),
            }},
            "sar": None if not sar else {
                "adi_pre": None if not np.isfinite(sar["adi_pre"][j, i]) else round(float(sar["adi_pre"][j, i]), 3),
                "adi_post": None if not np.isfinite(sar["adi_post"][j, i]) else round(float(sar["adi_post"][j, i]), 3),
                "z": None if sar["z"] is None else round(float(sar["z"][j, i]), 2)},
            "series": {k: [None if not np.isfinite(v) else round(float(v), 4)
                           for v in cells[k][:, j, i]] for k in cells},
            "chips": {"before": f"/chips/{run_id}/c{cid}_before.png",
                      "after": f"/chips/{run_id}/c{cid}_after.png"},
        })

    # ========================================================= CONFIRM ===
    # Only the ranked shortlist enters confirmation. The adapter makes a
    # conservative decision from evidence already attached to each candidate.
    cue = CUEResult(
        ranked_candidates=dets, retained=len(dets),
        evidence={
            "temporal": "harmonic residuals, max-t changepoint, permutation null, BH",
            "spectral": sorted(ix), "sar": bool(sar), "ranking": "reciprocal rank fusion",
        },
        explanation=mode,
    ).model_dump()
    confirmation = dependencies.confirmation.confirm(dets, say)
    confirm = CONFIRMResult.model_validate({**confirmation, "stage": "CONFIRM"}).model_dump()
    chain.add("report", {"detections": len(dets),
                         "certified": sum(d["certified"] for d in dets)})

    return {
        "ok": True, "run_id": run_id, "took": round(time.time() - t0, 1),
        "query": q, "window": [start, end], "bbox": list(bbox),
        "semantic": semantic.model_dump(exclude_none=False),
        "grid": [ny, nx], "alpha": alpha, "bh_threshold": thr,
        "composite": mode,
        "dates": dates, "model_dates": kept_dates, "split_date": split_date,
        "sources": per_source, "scenes_used": len(dates),
        "sar": None if not sar else {"scenes": sar["n"], "first": sar["dates"][0],
                                     "last": sar["dates"][-1]},
        "candidates": len(items),
        "tip": tip,
        "cue": cue,
        "confirm": confirm,
        "final_results": dets,
        "null": {"n": int(null.size), "p95": float(np.percentile(null, 95)),
                 "p99": float(np.percentile(null, 99))},
        "detections": dets,
        "images": {"before": f"/chips/{run_id}/before.png",
                   "after": f"/chips/{run_id}/after.png"},
        "stac": dependencies.provenance.stac_refs(used),
        "confirmation": confirmation,
        "provenance": chain.export(),
    }


def run_request(request: RunRequest, dependencies: PipelineDependencies | None = None,
                progress=None) -> dict:
    """Typed application-service entry point; external API payload stays unchanged."""
    return run(tuple(request.bbox), request.query, months=request.months,
               alpha=request.alpha, sources=tuple(request.sources), use_sar=request.use_sar,
               max_scenes=request.max_scenes, cloud=request.cloud, job=progress,
               dependencies=dependencies)
