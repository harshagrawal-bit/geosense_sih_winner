"""STAC provenance and a tamper-evident hash chain.

Each processing step is hashed together with the hash of the step before it,
so altering any recorded input or parameter invalidates every subsequent link.
Verification needs nothing but the chain itself - no server, no network.
"""
from __future__ import annotations
import hashlib, json, datetime as dt
from typing import Any, Dict, List

GENESIS = "0" * 64


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode()


class Chain:
    def __init__(self):
        self.links: List[Dict[str, Any]] = []
        self.head = GENESIS

    def add(self, step: str, detail: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"step": step, "detail": detail,
                   "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "prev": self.head}
        h = hashlib.sha256(self.head.encode() + _canon(payload)).hexdigest()
        link = {**payload, "hash": h, "n": len(self.links) + 1}
        self.links.append(link)
        self.head = h
        return link

    def verify(self) -> bool:
        prev = GENESIS
        for l in self.links:
            body = {k: l[k] for k in ("step", "detail", "ts", "prev")}
            if l["prev"] != prev:
                return False
            if hashlib.sha256(prev.encode() + _canon(body)).hexdigest() != l["hash"]:
                return False
            prev = l["hash"]
        return True

    def export(self):
        return {"head": self.head, "verified": self.verify(), "links": self.links}


def stac_refs(items):
    """Minimal STAC provenance record for the scenes actually used."""
    return [{"id": i["id"], "collection": i["collection"],
             "datetime": i["datetime"], "cloud": i["cloud"],
             "platform": i.get("platform", ""), "href": i.get("stac", "")}
            for i in items]
