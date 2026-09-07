"""Phase 1 regression checks for boundaries that do not need live network data."""
from __future__ import annotations

import time
import unittest
from dataclasses import replace

import numpy as np

from app import pipeline
from app.adapters import (DuckDbStorageAdapter, EvidenceConfirmationAdapter,
                          PostgreSQLStorageAdapter, ThreadPoolJobAdapter)
from app.repositories import DuckDBRunRepository, PostgreSQLRunRepository
from app.main import app
from app.pipeline import default_dependencies
from app.schemas import (CONFIRMResult, CUEResult, RunRequest,
                          SemanticRetrievalResult, TIPResult)


class PhaseOneContractsTests(unittest.TestCase):
    def test_existing_run_request_shape_is_valid(self):
        request = RunRequest(bbox=[77.54, 28.13, 77.62, 28.20], query="new construction")
        self.assertEqual(request.sources, ["sentinel-2-l2a"])

    def test_semantic_retrieval_returns_typed_rule_based_result(self):
        raw = default_dependencies().semantic.retrieve("new construction near roads")
        result = SemanticRetrievalResult.model_validate(raw)
        self.assertEqual(result.backend, "rule_based")
        self.assertFalse(result.model_available)
        self.assertIsNone(result.embedding)
        self.assertIn("construction", result.concepts)
        self.assertIn("spectral", result.evidence_types)

    def test_semantic_backend_selection_is_deterministic(self):
        adapter = default_dependencies().semantic
        self.assertEqual(adapter.retrieve("new construction"),
                         adapter.retrieve("new construction"))

    def test_pipeline_sends_raw_query_to_semantic_retrieval_before_tip(self):
        base = default_dependencies()
        calls = []

        class RecordingSemantic:
            def retrieve(self, text):
                calls.append(text)
                return base.semantic.retrieve(text)

        class EmptyCatalog:
            def search(self, *args, **kwargs):
                return []

            def coverage(self, items):
                return {"n": len(items)}

        result = pipeline.run(
            [77.54, 28.13, 77.62, 28.20], "new construction near roads",
            dependencies=replace(base, semantic=RecordingSemantic(), catalog=EmptyCatalog()),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(calls, ["new construction near roads"])

    def test_rule_and_temporal_baselines_remain_available_through_ports(self):
        dependencies = default_dependencies()
        parsed = dependencies.semantic.parse("deforestation and forest clearing")
        self.assertEqual(parsed["signature"], "veg_loss")
        residuals = np.array([[0.0], [0.0], [0.0], [0.0], [0.8], [0.9], [1.0], [1.0]])
        z, split, _ = dependencies.temporal.multiscale_scan(residuals, 1, 1, sign=1)
        self.assertGreater(float(z[0]), 0.0)
        self.assertGreaterEqual(int(split[0]), 3)

    def test_confirmation_evaluates_only_the_cue_shortlist(self):
        candidate = {"id": "cue-1", "certified": True,
                     "confirmation": {"targets": {"ndvi": -1.0, "ndbi": 1.0},
                                       "temporal": {"z": 3.0, "p": 0.01, "magnitude": 0.2}},
                     "deltas": {"ndvi": -0.2, "ndbi": 0.1}}
        result = EvidenceConfirmationAdapter().confirm([candidate], lambda *_: None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["shortlisted"], 1)
        self.assertEqual(result["evaluated"], 1)
        self.assertEqual(result["confirmed"], 1)
        self.assertEqual([item["id"] for item in result["evaluations"]], ["cue-1"])

    def test_confirmation_is_deterministic(self):
        candidate = {"id": "cue-1", "certified": True,
                     "confirmation": {"targets": {"ndvi": -1.0, "ndbi": 1.0},
                                       "temporal": {"z": 3.0, "p": 0.01, "magnitude": 0.2}},
                     "deltas": {"ndvi": -0.2, "ndbi": 0.1}}
        adapter = EvidenceConfirmationAdapter()
        self.assertEqual(adapter.confirm([candidate], lambda *_: None),
                         adapter.confirm([candidate], lambda *_: None))

    def test_confirmation_does_not_false_confirm_insufficient_or_unavailable(self):
        insufficient = {"id": "weak", "certified": False,
                        "confirmation": {"targets": {"ndvi": -1.0},
                                          "temporal": {"z": 1.0, "p": 0.4, "magnitude": 0.1}},
                        "deltas": {"ndvi": -0.1}}
        unavailable = {"id": "missing", "certified": True,
                       "confirmation": {"targets": {"ndvi": -1.0, "ndbi": 1.0},
                                         "temporal": {}}, "deltas": {}}
        result = EvidenceConfirmationAdapter().confirm([insufficient, unavailable], lambda *_: None)
        self.assertEqual(result["confirmed"], 0)
        self.assertEqual(result["insufficient"], 1)
        self.assertEqual(result["unavailable"], 1)
        self.assertEqual({item["status"] for item in result["evaluations"]},
                         {"insufficient_evidence", "unavailable_evidence"})

    def test_confirmation_has_no_legacy_unconfigured_state(self):
        result = default_dependencies().confirmation.confirm([], lambda *_: None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["confirmed"], 0)

    def test_tip_cue_confirm_stage_models_preserve_real_stage_contract(self):
        tip = TIPResult(candidates=24, screened=24, sources={}, evidence={})
        candidate = {"id": "run-0001", "certified": True}
        cue = CUEResult(ranked_candidates=[candidate], retained=1,
                        evidence={"temporal": "real"}, explanation="NDVI")
        confirmation = default_dependencies().confirmation.confirm([candidate], lambda *_: None)
        confirm = CONFIRMResult.model_validate({**confirmation, "stage": "CONFIRM"})
        self.assertEqual(tip.screened, tip.candidates)
        self.assertEqual(cue.ranked_candidates, [candidate])
        self.assertEqual(confirm.shortlisted, cue.retained)
        self.assertEqual(confirm.confirmed_candidates, [])

    def test_duckdb_repository_reads_saved_prototype_history(self):
        repository = DuckDbStorageAdapter()
        self.assertGreaterEqual(repository.stats()["runs"], 1)
        self.assertGreaterEqual(len(repository.recent()), 1)

    def test_duckdb_storage_adapter_accepts_repository_dependency(self):
        repository = DuckDBRunRepository()
        adapter = DuckDbStorageAdapter(repository)
        self.assertEqual(adapter.stats(), repository.stats())

    def test_postgres_repository_is_optional_skeleton(self):
        repository = PostgreSQLRunRepository("postgresql://unused")
        with self.assertRaises(NotImplementedError):
            repository.stats()
        with self.assertRaises(NotImplementedError):
            PostgreSQLStorageAdapter(repository).recent()

    def test_thread_pool_job_port_preserves_submit_and_poll_contract(self):
        jobs = ThreadPoolJobAdapter(max_workers=1)
        job_id = jobs.submit({"query": "test"}, lambda _id, progress: (progress("cue", 50, "working") or {"ok": True}))
        deadline = time.time() + 3
        status = jobs.status(job_id)
        while status and status["state"] == "running" and time.time() < deadline:
            time.sleep(0.02)
            status = jobs.status(job_id)
        self.assertIsNotNone(status)
        self.assertEqual(status["state"], "done")
        self.assertEqual(status["result"], {"ok": True})

    def test_existing_api_endpoints_remain_available(self):
        paths = {route.path for route in app.routes}
        self.assertTrue({"/api/run", "/api/job/{jid}", "/api/runs", "/api/sources"}.issubset(paths))


if __name__ == "__main__":
    unittest.main()
