# Inside GeoSense — overview

SIH26227 · Team VARCHASVA. What was built, what it runs on, and which parts are
real versus stand-ins.

---

## Two separate things in this folder

| | `index.html` | `app/` (run with `./run.sh`) |
|---|---|---|
| **Data** | scripted fixtures | live satellite archives |
| **What it is** | the pitch demo | the working prototype |
| **Opens** | double-click, no internet | `./run.sh` → http://127.0.0.1:8008 |
| **Detections** | 6 seeded examples; scenes drawn in JavaScript | real, computed per query |
| **Speed** | instant | 60–330 s per run |

Do not mix these up on stage.

---

## How a query becomes a detection

```
   Sentinel-2 L2A        Landsat C2 L2        Sentinel-1 RTC
   (Earth Search/AWS)    (Planetary Comp.)    (Planetary Comp.)
          |                    |                    |
          +--------------------+--------------------+
                               |  bbox + dates
                               v
                     +---------------------+
                     |     catalog.py      |  searches the archive
                     +---------------------+
                               |  ~26 scenes
                               v
                     +---------------------+
   query.py  ----->  |     raster.py       |  downloads ONLY your box,
   your words        |                     |  masks cloud, computes
   -> index weights  |                     |  NDVI NDBI MNDWI BSI
                     +---------------------+
                               |  256 grid cells, each a time series
                               v
                     +---------------------+
                     |     change.py       |  removes the seasonal cycle,
                     |                     |  finds the break, builds a
                     |                     |  null by shuffling time
                     +---------------------+  ----> provenance.py (hash chain)
                               |  a score and p-value per cell
                               v
                     +---------------------+
                     |      rank.py        |  fuses evidence, controls
                     +---------------------+  false discoveries
                               |               ----> store.py (DuckDB)
                               v
                  main.py (FastAPI) + app.html
                  map, ranked list, before/after chips
```

Nothing is pre-downloaded. Every run searches the live archives and streams
only the pixels inside your box.

---

## The models — read this before the viva

**There is now one neural network: RemoteCLIP.** It is optional (a checkbox in
the UI) because it needs ~1.5 GB of RAM. Everything else is classical
statistics — all real, all verifiable.

| Model | What it does |
|---|---|
| **RemoteCLIP ViT-B/32** | CLIP fine-tuned on remote-sensing image–text pairs. **Pretrained — we train nothing.** It scores each candidate on the *change* in similarity to your words: `cos(after, query) − cos(before, query)`. Absolute similarity would just rank cities that were already there. |

Only the shortlist the change detector surfaces gets embedded — 858 ms/chip on
this CPU means all 256 cells would take seven minutes — so tip-and-cue is
applied to the expensive *model* the same way it is applied to the expensive
*sensor*. SAM and TerraMind are still not included; they need more RAM than
this machine has.

The classical layer, unchanged:

| What | Does what, in plain words |
|---|---|
| **Spectral indices** | NDVI (vegetation), NDBI (built-up), MNDWI (water), BSI (bare soil) — band ratios that turn colour into a physical quantity |
| **Harmonic model** | Fits the yearly green-up/dry-down cycle per cell and subtracts it, so seasons aren't mistaken for change |
| **Changepoint scan** | Tries every possible split date, keeps the one where before and after differ most |
| **Permutation test** | Shuffles the dates to see how big a "change" pure noise can fake — that becomes the yardstick |
| **Benjamini–Hochberg** | 256 cells are tested at once, so this caps the share of flagged cells expected to be wrong at your α |
| **Reciprocal Rank Fusion** | Merges the separate rankings (signature match, change strength, radar) into one list |
| **Amplitude Dispersion** | Radar measure: low = stable hard surface. Works through monsoon cloud |
| **Query parser** | Rules. "construction" → expect NDVI to fall, NDBI to rise. The UI shows which rules fired. Runs always; RemoteCLIP re-ranks on top of it |

---

## What it runs on

| Layer | Used | Instead of |
|---|---|---|
| API | FastAPI + uvicorn | — |
| Imagery | rasterio / GDAL, pystac-client | — |
| Maths | NumPy only | — |
| Database | DuckDB | PostgreSQL + PostGIS + pgvector |
| Job queue | Python thread pool | Celery + Redis |
| Map UI | MapLibre GL (vendored, offline) | React + deck.gl |

The two substitutions are because this laptop has no Docker, no PostgreSQL and
no passwordless sudo. Both are one-module swaps — nothing above them depends on
the engine.

---

## Measured accuracy

We inject change of a known size into a **real** Sentinel-2 stack (23 dates over
Gurugram, real noise, real seasons, real cloud gaps) and measure how often the
detector finds it. Public benchmarks like OSCD and LEVIR-CD are *bi-temporal* —
two dates and a mask — so they cannot test a detector that needs a 15+ date
time series; scoring against them would test a different algorithm.

| Injected ΔNDVI | Recall | Precision | Empirical FDR |
|---|---|---|---|
| none | — | — | **0 cells flagged in 8 repeats** |
| 0.04 | 0% | — | 0% |
| 0.06 | 27% | 100% | 0% |
| **0.08** | **80%** | **100%** | **0%** |
| 0.12 | 100% | 100% | 0% |
| 0.24 | 100% | 100% | 0% |

**Minimum detectable change: ΔNDVI ≈ 0.08 at 80% power.** Empirical false
discovery rate 0% against a nominal α of 10% — the Benjamini–Hochberg guarantee
holds, conservatively. Reproduce with `python run_eval.py`.

## What it has actually produced

| Test | Result |
|---|---|
| Jewar, UP — vegetation gain | **2 certified** detections, p = 3.9 × 10⁻⁵ |
| Jewar, UP — construction | 0 certified; top candidate z = 2.07, radar ADI 0.365 → 0.251 |
| Simlipal, Odisha | 11 of 28 dates usable (monsoon cloud); 0 certified |
| Sentinel-1 tier | 14 RTC scenes fetched, ADI computed, hash chain verified |
| Semantic tier | RemoteCLIP loaded, 4 chip pairs embedded, re-ranked; delta spread −0.007…+0.043 |

> **If a run returns zero certified detections, that is a real answer, not a bug.**
> With ~15 usable dates and 256 simultaneous tests, only large clean changes clear
> FDR control at α = 0.10. Candidates below the line are still listed with their
> p-values. Saying this out loud is stronger than pretending otherwise.

---

## One bug worth mentioning if asked

The satellite catalogue tells you to subtract 0.1 from Sentinel-2 reflectance.
Applying it to this collection is **wrong** — over 0%-cloud Western Ghats forest
it produced `blue = -0.067` and `NDVI = 2.145`, both physically impossible.
Without it: `blue = 0.033`, `NDVI = 0.818` — textbook dense forest.

It was silently corrupting every index while still drawing plausible-looking
maps, which is exactly why the pipeline is checked against known land cover
rather than eyeballed.

---

## Running it

```bash
./run.sh          # http://127.0.0.1:8008
```

- Pick a preset area (Jewar, Gurugram, Dholera, Simlipal, Bengaluru) or drag a box on the map.
- Click a query chip, then **Run analysis**. Start it *before* you need it — it is genuinely downloading satellite data.
- Tick **Sentinel-1** to add the radar evidence stream.
- Tick **Semantic re-rank** to bring RemoteCLIP in. First run downloads ~600 MB and needs ~1.5 GB RAM; close Chrome first on this laptop.
- `python run_eval.py` reproduces the accuracy table above.

Full detail and every honest limitation: `README.md`. Demo running order: `DEMO.md`.
