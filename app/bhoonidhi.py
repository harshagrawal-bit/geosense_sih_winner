"""NRSC/ISRO Bhoonidhi client.

Bhoonidhi is the only Indian-operated source in the stack, and unlike
Copernicus or Planetary Computer it is not anonymous: every request needs a
registered account. The published contract at https://bhoonidhi-api.nrsc.gov.in
is a short-lived bearer token exchanged for a userId/password pair:

    POST /auth/token      userId + password -> access_token, refresh_token
    GET  /data/collections                  -> available product collections
    POST /data/search     spatial/temporal filters -> product metadata
    GET  /download                          -> the product itself

Credentials are read from the environment (BHOONIDHI_USER / BHOONIDHI_PASS)
and never stored in the repository or written to the run catalogue - a run's
provenance records which Bhoonidhi products were used, never who fetched them.
"""
from __future__ import annotations

import os
import time
import json
import urllib.parse
import urllib.request
import urllib.error

BASE = os.environ.get("BHOONIDHI_BASE", "https://bhoonidhi-api.nrsc.gov.in")
USER = os.environ.get("BHOONIDHI_USER")
PASS = os.environ.get("BHOONIDHI_PASS")
TIMEOUT = 30

_token: dict = {"access": None, "refresh": None, "expires": 0.0}


def configured() -> bool:
    return bool(USER and PASS)


def _post(path, payload, headers=None, form=False):
    if form:
        body = "&".join(f"{k}={urllib.parse.quote(str(v))}"
                        for k, v in payload.items()).encode()
        ctype = "application/x-www-form-urlencoded"
    else:
        body = json.dumps(payload).encode()
        ctype = "application/json"
    req = urllib.request.Request(
        BASE + path, data=body, method="POST",
        headers={"Content-Type": ctype, "Accept": "application/json",
                 **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode() or "{}")


def _get(path, headers=None, params=None):
    url = BASE + path
    if params:
        url += "?" + "&".join(f"{k}={urllib.parse.quote(str(v))}"
                              for k, v in params.items())
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode() or "{}")


def token(force=False) -> str | None:
    """Fetch (and cache) an access token. Returns None if unconfigured."""
    if not configured():
        return None
    if not force and _token["access"] and time.time() < _token["expires"] - 60:
        return _token["access"]

    creds = {"userId": USER, "password": PASS}
    last = None
    # The spec names the fields but not the encoding; try the documented JSON
    # shape first, then form-encoding, rather than guessing once and failing.
    for form in (False, True):
        try:
            r = _post("/auth/token", creds, form=form)
            acc = r.get("access_token") or r.get("accessToken") or r.get("token")
            if acc:
                _token.update(
                    access=acc,
                    refresh=r.get("refresh_token") or r.get("refreshToken"),
                    expires=time.time() + float(r.get("expires_in", 900)))
                return acc
            last = f"no token field in response: {list(r)[:6]}"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:180].decode(errors='replace')}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"Bhoonidhi auth failed - {last}")


def _auth_headers():
    t = token()
    if not t:
        raise RuntimeError("Bhoonidhi credentials not set "
                           "(BHOONIDHI_USER / BHOONIDHI_PASS)")
    return {"Authorization": f"Bearer {t}"}


def collections():
    return _get("/data/collections", headers=_auth_headers())


def search(bbox, start: str, end: str, collection: str | None = None,
           limit: int = 50):
    """bbox is (west, south, east, north) in EPSG:4326."""
    w, s, e, n = bbox
    payload = {"bbox": [w, s, e, n], "startDate": start, "endDate": end,
               "limit": limit}
    if collection:
        payload["collection"] = collection
    return _post("/data/search", payload, headers=_auth_headers())


def status() -> dict:
    """Non-throwing health check for the UI."""
    if not configured():
        return {"configured": False,
                "hint": "set BHOONIDHI_USER and BHOONIDHI_PASS in .env"}
    try:
        token(force=True)
        return {"configured": True, "authenticated": True, "base": BASE}
    except Exception as ex:
        return {"configured": True, "authenticated": False,
                "error": str(ex)[:240]}
