# GeoSense — what is in this folder

Two separate things. Do not confuse them on stage.

| | file | data | use it for |
|---|---|---|---|
| **Pitch demo** | `index.html` | scripted fixtures | the 12-minute talk; opens instantly, always works |
| **Working prototype** | `./run.sh` → `app/` | live Sentinel-2 / Landsat / Sentinel-1 | proving it is real |

---

## A · Pitch demo — `index.html`

Double-click it. No server, no internet. Six seeded detections across India.
Run order keyed to your slides:

1. **Slide 2.** Hit **Search**. The funnel counts 2,412,886 → 312 → 6 in 4.9 s.
   Click a chip to re-run with different wording.
2. **Slide 3.** Detection #1 → **Imagery** tab (before/after with detection
   boxes), then **Time-Series** (COLD harmonic fit, ADI). Right panel is the
   hash-chained provenance.
3. **Slide 4.** Detection **#5 Kosi basin** — optical is 94 % cloud, banner says
   *SAR USED*. Detection **#6 Jharkhand** — 79.9 %, "below the conformal
   acceptance floor, routed to the human queue". Your risk table, demonstrated.
4. **Slide 5.** Back to **Map** — six detections from one query.

Keys `1`/`2`/`3` switch tabs, `↑`/`↓` walk the queue.

**The six detections are seeded fixtures, not live inference.** If a judge asks,
say so — then open the prototype.

---

## B · Working prototype — `./run.sh`

```bash
./run.sh          # http://127.0.0.1:8008
```

This one is real: live STAC search, real COG pixels, real statistics.
A run takes **60–200 s**, so start it *before* you need it.

Suggested live demo:

1. Preset **Jewar, UP** → query chip **construction** → **Run analysis**.
   Narrate the progress line: it is genuinely streaming COG windows.
2. When it lands, point at **Run summary**: scenes used, null draws, the BH
   threshold, certified count. Then a detection → before/after chips cut at
   *that cell's own* break date, and the index time series.
3. Tick **Add Sentinel-1 SAR tier** and re-run to show the radar evidence
   (ADI before → after) joining the fusion.

If a run returns **zero certified detections**, that is not a failure — say so.
With ~15 usable dates and 256 simultaneous tests, only large clean changes clear
FDR control at α = 0.10. Candidates below threshold are still listed with their
p-values. Raising α in the UI shows the trade-off honestly.

**Failure modes you should be ready for**, all of which the UI reports plainly:
cloud-dominated AOI (Simlipal in monsoon can lose most of the stack), an AOI
drawn too small or too large, and a slow venue network — the bottleneck is
network I/O, not compute.

See `README.md` for what is real, what is substituted, and why.
