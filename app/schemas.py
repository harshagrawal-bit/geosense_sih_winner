"""Typed API and domain contracts for the current prototype.

These schemas describe the product as it exists today.  They intentionally do
not imply a future database, queue, embedding model, or change model.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class AOI(BaseModel):
    """A geographic bounding box in WGS84: west, south, east, north."""
    bbox: list[float] = Field(..., min_length=4, max_length=4)

    @field_validator("bbox")
    @classmethod
    def finite_coordinates(cls, value: list[float]) -> list[float]:
        if not all(float("-inf") < coordinate < float("inf") for coordinate in value):
            raise ValueError("bbox values must be finite")
        return value

    @model_validator(mode="after")
    def valid_extent(self) -> "AOI":
        west, south, east, north = self.bbox
        if not (east > west and north > south):
            raise ValueError("bbox must be [west, south, east, north]")
        return self


class QueryRequest(BaseModel):
    query: str = "new construction"
    months: int | None = Field(default=None, ge=3, le=120)
    alpha: float = Field(default=0.10, gt=0, le=0.5)
    cloud: int = Field(default=45, ge=0, le=100)
    max_scenes: int = Field(default=24, ge=8, le=40)
    sources: list[str] = Field(default_factory=lambda: ["sentinel-2-l2a"])
    use_sar: bool = False


class RunRequest(AOI, QueryRequest):
    """The unchanged request body accepted by POST /api/run."""


class SemanticRetrievalResult(BaseModel):
    query: str
    normalized_query: str
    concepts: list[str]
    evidence_types: list[str]
    backend: Literal["rule_based"]
    model_available: bool = False
    embedding: list[float] | None = None
    intent: dict[str, Any]


class TIPResult(BaseModel):
    stage: Literal["TIP"] = "TIP"
    candidates: int = Field(ge=0)
    screened: int = Field(ge=0)
    sources: dict[str, Any]
    evidence: dict[str, Any]


class CUEResult(BaseModel):
    stage: Literal["CUE"] = "CUE"
    ranked_candidates: list[dict[str, Any]]
    retained: int = Field(ge=0)
    evidence: dict[str, Any]
    explanation: str


class CONFIRMResult(BaseModel):
    stage: Literal["CONFIRM"] = "CONFIRM"
    status: Literal["completed"]
    shortlisted: int = Field(ge=0)
    evaluated: int = Field(ge=0)
    confirmed: int = Field(ge=0)
    insufficient: int = Field(ge=0)
    unavailable: int = Field(ge=0)
    confirmed_candidates: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]
    evidence: dict[str, Any]
    message: str


class SceneMetadata(BaseModel):
    id: str
    source: str
    collection: str
    datetime: str
    cloud: float = 0.0
    platform: str = ""
    stac: str = ""


class ProvenanceLink(BaseModel):
    n: int
    step: str
    detail: dict[str, Any]
    ts: str
    prev: str
    hash: str


class ProvenanceRecord(BaseModel):
    head: str
    verified: bool
    links: list[ProvenanceLink]


class Evidence(BaseModel):
    z: float
    p: float
    confidence: float
    deltas: dict[str, float]
    sar: dict[str, float | None] | None = None


class Detection(BaseModel):
    id: str
    cell: list[int]
    bbox: list[float]
    z: float
    p: float
    confidence: float
    certified: bool
    break_date: str
    magnitude: float
    deltas: dict[str, float]
    chips: dict[str, str]


class RankingResult(BaseModel):
    bh_threshold: float
    alpha: float
    detections: list[Detection]


class JobStatus(BaseModel):
    id: str
    state: Literal["running", "done", "error"]
    pct: int = Field(ge=0, le=100)
    stage: str
    message: str
    request: dict[str, Any]
    result: dict[str, Any] | None = None


class ConfirmationResult(BaseModel):
    """Compatibility shape for a completed evidence confirmation result."""
    status: Literal["completed"] = "completed"
    shortlisted: int = Field(ge=0)
    evaluated: int = Field(ge=0)
    confirmed: int = Field(ge=0)
    insufficient: int = Field(ge=0)
    unavailable: int = Field(ge=0)
    message: str
