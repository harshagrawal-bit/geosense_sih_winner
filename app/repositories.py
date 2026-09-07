"""Persistence repositories used by the application service.

The repository contract keeps persistence concerns out of the analysis
pipeline. DuckDB is the complete local implementation; PostgreSQL is an
optional seam for a later deployment and deliberately has no driver import.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Protocol

import duckdb

from .config import DB_PATH


class RunRepository(Protocol):
    def save(self, result: dict[str, Any]) -> None: ...
    def recent(self, limit: int = 25) -> list[dict[str, Any]]: ...
    def stats(self) -> dict[str, int]: ...


_DDL = """
CREATE TABLE IF NOT EXISTS runs (
  run_id VARCHAR PRIMARY KEY, created TIMESTAMP, query VARCHAR,
  signature VARCHAR, bbox DOUBLE[], win_start DATE, win_end DATE,
  alpha DOUBLE, scenes_used INTEGER, candidates INTEGER,
  bh_threshold DOUBLE, n_detections INTEGER, n_certified INTEGER,
  took DOUBLE, chain_head VARCHAR, payload JSON);
CREATE TABLE IF NOT EXISTS detections (
  det_id VARCHAR PRIMARY KEY, run_id VARCHAR, cell_y INTEGER, cell_x INTEGER,
  bbox DOUBLE[], z DOUBLE, p DOUBLE, confidence DOUBLE, certified BOOLEAN,
  break_date DATE, magnitude DOUBLE, deltas JSON);
CREATE TABLE IF NOT EXISTS scenes (
  run_id VARCHAR, stac_id VARCHAR, collection VARCHAR, dt VARCHAR,
  cloud DOUBLE, platform VARCHAR, href VARCHAR);
CREATE TABLE IF NOT EXISTS provenance (
  run_id VARCHAR, n INTEGER, step VARCHAR, ts VARCHAR,
  prev VARCHAR, hash VARCHAR, detail JSON);
"""


class DuckDBRunRepository:
    """Working embedded repository for local runs."""

    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._lock = threading.Lock()

    def _connection(self):
        connection = duckdb.connect(self.path)
        connection.execute(_DDL)
        return connection

    def save(self, result: dict[str, Any]) -> None:
        if not result.get("ok"):
            return
        with self._lock, self._connection() as connection:
            run_id = result["run_id"]
            for table in ("runs", "detections", "scenes", "provenance"):
                connection.execute(f"DELETE FROM {table} WHERE run_id=?", [run_id])
            connection.execute(
                "INSERT INTO runs VALUES (?,now(),?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [run_id, result["query"]["raw"], result["query"]["signature"],
                 result["bbox"], result["window"][0], result["window"][1], result["alpha"],
                 result["scenes_used"], result["candidates"], result["bh_threshold"],
                 len(result["detections"]), sum(d["certified"] for d in result["detections"]),
                 result["took"], result["provenance"]["head"], json.dumps({
                     "query": result["query"], "null": result["null"],
                     "composite": result["composite"], "images": result["images"],
                     "model_dates": result["model_dates"]})])
            for detection in result["detections"]:
                connection.execute("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                    detection["id"], run_id, detection["cell"][0], detection["cell"][1],
                    detection["bbox"], detection["z"], detection["p"], detection["confidence"],
                    detection["certified"], detection["break_date"], detection["magnitude"],
                    json.dumps(detection["deltas"])])
            for scene in result["stac"]:
                connection.execute("INSERT INTO scenes VALUES (?,?,?,?,?,?,?)", [
                    run_id, scene["id"], scene["collection"], scene["datetime"],
                    scene["cloud"], scene["platform"], scene["href"]])
            for link in result["provenance"]["links"]:
                connection.execute("INSERT INTO provenance VALUES (?,?,?,?,?,?,?)", [
                    run_id, link["n"], link["step"], link["ts"], link["prev"],
                    link["hash"], json.dumps(link["detail"], default=str)])

    def recent(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT run_id,created,query,signature,scenes_used,n_detections,"
                "n_certified,took FROM runs ORDER BY created DESC LIMIT ?", [limit]).fetchall()
        return [{"run_id": row[0], "created": str(row[1]), "query": row[2],
                 "signature": row[3], "scenes": row[4], "detections": row[5],
                 "certified": row[6], "took": row[7]} for row in rows]

    def stats(self) -> dict[str, int]:
        with self._lock, self._connection() as connection:
            return {
                "runs": connection.execute("SELECT count(*) FROM runs").fetchone()[0],
                "detections": connection.execute("SELECT count(*) FROM detections").fetchone()[0],
                "scenes": connection.execute("SELECT count(DISTINCT stac_id) FROM scenes").fetchone()[0],
            }


class PostgreSQLRunRepository:
    """Future PostGIS repository boundary; unavailable in the local prototype."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn

    def _unavailable(self) -> None:
        raise NotImplementedError(
            "PostgreSQL/PostGIS storage is an architectural skeleton; "
            "configure DuckDB for the local prototype")

    def save(self, result: dict[str, Any]) -> None:
        self._unavailable()

    def recent(self, limit: int = 25) -> list[dict[str, Any]]:
        self._unavailable()

    def stats(self) -> dict[str, int]:
        self._unavailable()