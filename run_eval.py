"""Run the semi-synthetic validation and write a report."""
import json, sys, time
import numpy as np
from app import evaluate as ev

BBOX = (77.00, 28.35, 77.10, 28.45)          # Gurugram - mixed urban/farmland
CACHE = "data/eval_gurugram.npz"
ALPHA = 0.10
MAGS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24]

t0 = time.time()
print("1. real Sentinel-2 stack over Gurugram", flush=True)
dates, cells = ev.fetch_cube(BBOX, "2024-01-01", "2026-08-31",
                             cloud=45, max_scenes=26, cache=CACHE)
ny, nx = cells["ndvi"].shape[1:]
print(f"   {len(dates)} dates  {dates[0]} -> {dates[-1]}  grid {ny}x{nx} "
      f"= {ny*nx} cells  ({time.time()-t0:.0f}s)", flush=True)

print("\n2. false-positive check - NO change injected, 8 repeats", flush=True)
null_rows = ev.sweep(cells, dates, [0.0], repeats=8, alpha=ALPHA)
fp = null_rows[0]["detected"]
print(f"   mean cells flagged on unchanged imagery: {fp:.2f} of {ny*nx}"
      f"   ({100*fp/(ny*nx):.2f}% of the grid)", flush=True)

print(f"\n3. power curve - injected change, 5 repeats each, alpha={ALPHA}",
      flush=True)
rows = ev.sweep(cells, dates, MAGS[1:], repeats=5, alpha=ALPHA)

mdc80 = ev.min_detectable(rows, 0.80)
mdc50 = ev.min_detectable(rows, 0.50)
print("\n4. summary", flush=True)
print(f"   minimum detectable change (50% power): dNDVI "
      f"{'%.3f' % mdc50 if mdc50 else 'not reached'}")
print(f"   minimum detectable change (80% power): dNDVI "
      f"{'%.3f' % mdc80 if mdc80 else 'not reached'}")
mean_fdr = float(np.mean([r["fdr"] for r in rows if r["recall"] > 0.2]))
print(f"   empirical FDR where detector fires: {100*mean_fdr:.1f}%  "
      f"(nominal alpha = {100*ALPHA:.0f}%)")

out = {"aoi": list(BBOX), "dates": dates, "grid": [ny, nx], "alpha": ALPHA,
       "null": null_rows, "sweep": rows,
       "min_detectable_50": mdc50, "min_detectable_80": mdc80,
       "empirical_fdr": mean_fdr, "seconds": round(time.time()-t0, 1)}
json.dump(out, open("data/evaluation.json", "w"), indent=1)
print(f"\nwrote data/evaluation.json  ({time.time()-t0:.0f}s total)")
