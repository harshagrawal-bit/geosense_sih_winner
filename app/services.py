"""Application services: API-facing use cases, not infrastructure details."""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

from .adapters import ThreadPoolJobAdapter, configured_storage_adapter
from .contracts import JobSubmissionPort, StoragePort
from . import pipeline
from .schemas import RunRequest


@dataclass
class AnalysisService:
    storage: StoragePort = field(default_factory=configured_storage_adapter)
    jobs: JobSubmissionPort = field(default_factory=ThreadPoolJobAdapter)
    dependencies: pipeline.PipelineDependencies = field(default_factory=pipeline.default_dependencies)

    def submit(self, request: RunRequest) -> str:
        payload = request.model_dump()

        def work(_job_id: str, progress):
            result = pipeline.run_request(request, dependencies=self.dependencies, progress=progress)
            if result.get("ok"):
                try:
                    self.storage.save(result)
                except Exception:
                    result["store_error"] = traceback.format_exc(limit=2)
            return result

        return self.jobs.submit(payload, work)

    def job_status(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.status(job_id)

    def recent_runs(self) -> dict[str, Any]:
        return {"runs": self.storage.recent(), "stats": self.storage.stats()}
