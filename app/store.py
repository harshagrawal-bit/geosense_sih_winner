"""Backward-compatible module functions for the local DuckDB repository."""
from __future__ import annotations

from .repositories import DuckDBRunRepository

_repository = DuckDBRunRepository()


def save(res: dict):
    _repository.save(res)


def recent(limit=25):
    return _repository.recent(limit)


def stats():
    return _repository.stats()
