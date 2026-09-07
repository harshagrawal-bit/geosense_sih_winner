# GeoSense — working prototype

Semantic retrieval and multi-temporal change analysis of satellite imagery.
SIH26227 · Team VARCHASVA.

This is a **working prototype**, not a mock-up. You draw an area, describe the
change you are looking for in plain English, and it searches the live public
satellite archives, streams only the pixels it needs, fits a seasonal model per
cell, and returns ranked change detections with calibrated confidence and a
verifiable provenance chain.

`index.html` in this folder is the separate, self-contained **pitch demo** with
scripted data. It is untouched. The prototype is everything under `app/`.

---

## Run it

```bash
./run.sh                      # http://127.0.0.1:8008
```

First time only:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then: pick a preset AOI (or drag one on the map), pick a query chip, **Run
analysis**. A run over a ~9 km box takes **60–200 s**, almost all of it
spent streaming COG windows over the network.

---

## What it actually does

### 1 · Discover — live archive search
`app/catalog.py` queries STAC APIs per request. Nothing is pre-staged.

| Source | Endpoint | Access |
|---|---|---|
| Copernicus **Sentinel-2 L2A** | Earth Search v1 (AWS `sentinel-cogs`) | anonymous |
| USGS **Landsat Collection 2 L2** | Microsoft Planetary Computer | free, auto-signed |
| Copernicus **Sentinel-1** RTC / GRD | Microsoft Planetary Computer | free, auto-signed |
| NRSC/ISRO **Bhuvan** | `bhuvan-vec2.nrsc.gov.in` WMS | basemap layer in the UI |

`app/raster.py` reads only the AOI window from each COG, at an overview level
chosen by the output shape. A 25-date stack costs a few MB, not a few GB.
Cloud and shadow are masked per pixel from Sentinel-2 SCL and Landsat
`QA_PIXEL`, so a partly clouded scene still contributes its clear pixels.

### 2 · Prioritise — seasonal model per cell
`app/change.py`. The AOI is read into 256×256 px and aggregated into a 16×16
grid of cells. For each cell and each index (NDVI, NDBI, MNDWI, BSI):

1. Fit a harmonic model of the annual cycle (1–2 harmonics) and subtract it,
   leaving residuals with phenology removed. **No linear trend term** — over a
   1–2 year window it is not identifiable and extrapolates catastrophically
   (this was measured, not assumed).
2. Combine indices into one composite in the direction the query implies, so a
   matching change is positive.
3. Box-filter the residual field 3×3. Real change covers adjacent cells;
   per-cell sensor noise does not.
4. Scan every split point and keep the largest two-sample t statistic.

Ranking fuses the signature score, the change statistic and the p-value with
**Reciprocal Rank Fusion**, plus context terms derived from the imagery itself
("near a river" is resolved from MNDWI, not from a gazetteer).

### 3 · Verify — calibrated confidence
Permuting the time order of the residuals destroys any real step while keeping
the noise distribution intact, so the resulting statistics are draws from the
no-change null **for this scene, this cadence, this noise level**. With 160
permutations over 160 cells that is ~25 600 null draws per run.

```
p = (1 + #{null ≥ observed}) / (n_null + 1)
```

Benjamini–Hochberg on those p-values then controls the **false discovery rate**
at the chosen α. A query naming a direction ("construction") is tested
**one-sided** — otherwise a change of the opposite kind would certify against it.

So "certified at α = 0.10" means: among the cells flagged, at most 10 % are
expected to be false, under a null calibrated from this scene's own noise.

### Provenance
`app/provenance.py` hash-chains every step (query → discover → ingest → model →
certify → report) with the STAC IDs of the scenes actually used. Altering any
recorded input invalidates every later link. Verification needs nothing but the
chain — no server, no network. Everything persists to DuckDB (`app/store.py`).

---

## Honest notes

**Read this before demoing.** These are real limitations, not modesty.

- **Semantic search is now two tiers, and the second is optional.** The rule
  parser in `app/query.py` always runs and maps a query to the physical
  signature the change produces (NDVI falls, NDBI rises, …); every rule is
  inspectable and the UI shows which fired. On top of that, ticking "Semantic
  re-rank" loads **RemoteCLIP ViT-B/32** (`app/semantic.py`) and reorders the
  candidate shortlist. It is off by default because it needs ~1.5 GB of RAM on
  a 5 GB machine, and it degrades back to the rule parser on any failure rather
  than refusing to run.
- **The semantic score is a delta, not a similarity.** Asking "does this look
  like construction?" ranks cities that were already there. Each candidate is
  embedded twice and scored `cos(after, q) - cos(before, q)`, so only places
  that *moved toward* the description rank up. CLIP also gets a wider crop
  (80 px) than the evidence panel (32 px): the tight chip that suits a human
  reviewer starves a model trained on whole scenes, and widening it lifted the
  score spread from ±0.004 to -0.007..+0.043.
- **Semantic power is still limited by chip resolution.** At GRID_PX=256 a cell
  is ~24 real pixels. The next real improvement is reading the AOI at higher
  resolution for the chip pass; that is not done.
- **Substituted infrastructure.** The architecture calls for PostgreSQL 16 +
  PostGIS 3.4 + pgvector and Celery + Redis. This machine has no PostgreSQL, no
  Docker, no Redis and no passwordless sudo, so the prototype uses **DuckDB**
  with the same logical schema and a **bounded thread pool** with the same
  submit/poll contract. Both are one-module swaps.
- **Change detection is COLD-inspired, not COLD.** It uses the same idea —
  harmonic seasonal model, break as unexplained deviation — but it is a
  deseasonalise-then-changepoint scan, not the published CCDC/COLD algorithm.
  Do not claim otherwise.
- **The SAR tier is real but secondary.** Tick "Add Sentinel-1 SAR tier" and it
  fetches Sentinel-1 **RTC** from Planetary Computer, computes the Amplitude
  Dispersion Index before and after the break, runs an independent backscatter
  changepoint, and adds both as evidence streams to the RRF fusion. RTC is
  required, not a preference: GRD is not terrain-corrected, so pixels would not
  align between dates and ADI would be measuring geometry rather than ground.
  Certification still rests on the optical statistic; SAR informs ranking.
- **Statistical power is genuinely limited.** With ~15 usable dates and 256
  simultaneous tests, only large and clean changes clear BH at α = 0.10. A run
  reporting *zero certified detections* is usually telling the truth about the
  data. Candidates below the threshold are still listed with their p-values.
- **Cloud is the binding constraint over India.** Monsoon-season AOIs (Simlipal
  in July, say) can lose most of the stack. The pipeline relaxes its coverage
  threshold in steps and reports which one it used; if it still cannot fit a
  model it says so rather than inventing a result.
- **Basemap tiles need internet.** MapLibre itself is vendored locally, but Esri
  / OSM / Bhuvan tiles are fetched live. True air-gapped operation needs a local
  tile store; nothing else in the pipeline requires the public internet once
  imagery is cached.

## Measured accuracy

`python run_eval.py` injects change of known size into a real 23-date
Sentinel-2 stack over Gurugram and measures detection rate. It calls the same
`change.*` routines the pipeline calls, so it measures the shipped detector.

| Injected ΔNDVI | Recall | Precision | Empirical FDR |
|---|---|---|---|
| none (8 repeats) | — | — | **0 cells flagged** |
| 0.06 | 27% | 100% | 0% |
| 0.08 | 80% | 100% | 0% |
| 0.12 | 100% | 100% | 0% |

Minimum detectable change **ΔNDVI ≈ 0.08 at 80% power**; empirical FDR 0%
against a nominal α of 10%. Results in `data/evaluation.json`.

Public bi-temporal benchmarks (OSCD, LEVIR-CD) are not used, and the reason is
not convenience: they supply two dates and a mask, whereas this detector needs
a 15+ date series to fit a seasonal model. Scoring against them would measure a
different algorithm.

## One measurement worth knowing about

Earth Search publishes `raster:bands` with `offset = -0.1` on Sentinel-2 L2A
assets, which is the documented baseline-04.00 DN correction. **Applying it to
this collection is wrong** - these COGs are already harmonised. Measured over
0 %-cloud Western Ghats forest:

| | blue | green | red | NIR | NDVI |
|---|---|---|---|---|---|
| offset applied | **-0.067** | -0.051 | -0.071 | 0.194 | **2.145** |
| no offset | 0.033 | 0.049 | 0.029 | 0.294 | **0.818** |

Negative reflectance is impossible and NDVI is bounded by 1, so the second row
is the physical one. `trust_stac_scale=False` in `app/config.py` records this.
It is the kind of bug that silently corrupts every index downstream while
still producing plausible-looking maps, which is why the pipeline is checked
against known land cover rather than eyeballed.

## Layout

```
app/catalog.py     STAC search across S2 / Landsat / S1
app/raster.py      windowed COG reads, cloud masking, indices, chips
app/change.py      harmonic model, changepoint scan, permutation null, ADI
app/query.py       natural language -> physical change signature (rules)
app/semantic.py    RemoteCLIP tier: delta-similarity re-ranking (optional)
app/evaluate.py    semi-synthetic validation harness
app/rank.py        RRF fusion, context priors, Benjamini-Hochberg
app/provenance.py  hash-chained audit trail
app/store.py       DuckDB catalogue
app/pipeline.py    tip-and-cue orchestration
app/main.py        FastAPI
app/static/app.html  analyst console (MapLibre, vendored)
```

## Storage boundary

Semantic retrieval currently uses the inspectable rule-based parser in
`app/query.py` through `SemanticRetrievalPort`. GeoRSCLIP and GeoLangBind are
not deployed: this environment has no Torch, Transformers, or local model
weights, and no model is downloaded automatically. The API and UI report
`rule_based` with `model_available: false`; no embeddings are fabricated.

The application uses the real DuckDB repository by default. The backend can be
selected without installing PostgreSQL:

```bash
GEOSENSE_STORAGE=duckdb                 # default local repository
GEOSENSE_STORAGE=postgres \
GEOSENSE_POSTGRES_DSN=postgresql://...  # interface skeleton only
```

`app/repositories.py` contains the repository contract and the working
DuckDB implementation. The PostgreSQL/PostGIS class is intentionally not
implemented yet and raises a clear error if selected, so local startup never
requires PostgreSQL.

## Phase 1 architecture boundaries

The live prototype still uses the implementations described above. Phase 1
adds typed contracts in `app/schemas.py`, replacement ports in
`app/contracts.py`, and adapters in `app/adapters.py`. The application path is
now `API → AnalysisService → pipeline → adapters → DuckDB`. The orchestrator
makes `TIP → CUE → CONFIRM` explicit: Confirm evaluates the CUE shortlist using
existing temporal and spectral evidence, with optional SAR evidence recorded
when available. No segmentation or VHR backend is present.

These boundaries do **not** mean that Celery, PostgreSQL, STAC provenance,
GeoRSCLIP, CCDC, or confirmation models are present. They only isolate the
current rule parser, harmonic residual/max-t detector, Sentinel-1 RTC/ADI
evidence, DuckDB store, thread-pool jobs, and hash-chain provenance so they
can be replaced independently later.
