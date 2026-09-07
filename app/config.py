"""Central configuration. No credentials required for the default sources."""
import os


def _load_dotenv(path=None):
    """Minimal .env loader - no dependency, and it never overrides a real
    environment variable, so a deployment can still set secrets properly."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()

# --- STAC endpoints (7.1 Primary Imagery Sources) -------------------------
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
PLANETARY    = "https://planetarycomputer.microsoft.com/api/stac/v1"

SOURCES = {
    # Copernicus Sentinel-2 L2A surface reflectance. Public COGs on AWS
    # (s3://sentinel-cogs), anonymous HTTPS access, no account needed.
    "sentinel-2-l2a": {
        "api": EARTH_SEARCH, "collection": "sentinel-2-l2a", "sign": False,
        "label": "Copernicus Sentinel-2 L2A", "gsd": 10,
        "bands": {"blue": "blue", "green": "green", "red": "red",
                  "nir": "nir", "swir16": "swir16", "swir22": "swir22",
                  "scl": "scl"},
        # DN/10000 IS surface reflectance for this collection - do not apply
        # the baseline-04.00 -1000 DN offset. Earth Search publishes
        # raster:bands offset=-0.1 on these assets, but applying it is wrong
        # here: measured over 0%-cloud Western Ghats forest it gives blue
        # =-0.067 and NDVI=2.145, which is impossible (NDVI is bounded by 1).
        # Without it the same pixels give blue=0.033, NIR=0.294, NDVI=0.818 -
        # textbook dense vegetation. Hence trust_stac_scale=False.
        "scale": 1e-4, "offset": 0.0, "trust_stac_scale": False,
    },
    # USGS Landsat Collection 2 Level-2. Via Planetary Computer (free, signed);
    # the AWS copy is requester-pays so we deliberately do not use it.
    "landsat-c2-l2": {
        "api": PLANETARY, "collection": "landsat-c2-l2", "sign": True,
        "label": "USGS Landsat Collection 2 L2", "gsd": 30,
        "bands": {"blue": "blue", "green": "green", "red": "red",
                  "nir": "nir08", "swir16": "swir16", "swir22": "swir22",
                  "scl": "qa_pixel"},
        "scale": 2.75e-5, "offset": -0.2,  # C2 L2 SR scaling
    },
    # Sentinel-1 SAR. RTC is terrain-corrected (pixels align across dates, so
    # Amplitude Dispersion is meaningful); GRD is the unrestricted fallback.
    "sentinel-1-rtc": {
        "api": PLANETARY, "collection": "sentinel-1-rtc", "sign": True,
        "label": "Copernicus Sentinel-1 RTC", "gsd": 10,
        "bands": {"vv": "vv", "vh": "vh"}, "sar": True,
    },
    "sentinel-1-grd": {
        "api": PLANETARY, "collection": "sentinel-1-grd", "sign": True,
        "label": "Copernicus Sentinel-1 GRD", "gsd": 10,
        "bands": {"vv": "vv", "vh": "vh"}, "sar": True,
    },
}

# NRSC/ISRO Bhuvan. Open WMS/WMTS layers are browsable without an account, but
# Bhoonidhi *downloads* (Resourcesat/Cartosat scenes) require a registered
# login, so we expose Bhuvan as a reference/basemap layer only. See README.
BHUVAN_WMS = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
DB_PATH   = os.path.join(DATA_DIR, "geosense.duckdb")
STORAGE_BACKEND = os.getenv("GEOSENSE_STORAGE", "duckdb").lower()
POSTGRES_DSN = os.getenv("GEOSENSE_POSTGRES_DSN")

# Analysis grid: AOI is read into GRID_PX x GRID_PX pixels, then aggregated
# into CELL x CELL blocks. 256/16 -> a 16x16 grid of 256 analysis cells.
GRID_PX = 256
CELL    = 16

MAX_SCENES   = 40     # per sensor, per request
CLOUD_LIMIT  = 60     # scene-level cloud cover % to even consider
os.makedirs(CACHE_DIR, exist_ok=True)
