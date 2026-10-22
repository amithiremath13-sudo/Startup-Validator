"""
tests/test_api.py
==================
Phase 6A — API endpoint tests.

Uses FastAPI's TestClient (backed by httpx) so tests run entirely
in-process — no server needs to be running.

TEST CASES
----------
1. Valid request           → 200 with well-structured response
2. Empty idea string       → 400
3. Whitespace-only idea    → 400
4. Missing "idea" field    → 422 (Pydantic validation)
5. Missing "country" field → 422 (Pydantic validation)
6. Empty country string    → 400
7. Unresolvable country    → 200 (pipeline handles gracefully, country_Unknown)
8. Scores are separate     → combined_score is null, scores_are_separate is true
9. ML score is not called a probability in the note
10. Simulated pipeline failure → 500 with safe error message (no stack trace)
"""

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

from api.app import app

# TestClient wraps the app in a context manager so the lifespan fires,
# loading the ML model once before any test runs.
# raise_server_exceptions=True (default) lets real errors surface in tests.
client = TestClient(app)


# Force the lifespan to run at module import time so the model is loaded
# before any test class's setUpClass runs.
# enter() starts the lifespan; the client tears it down in __exit__.
client.__enter__()


VALID_PAYLOAD = {
    "idea":    "An AI-powered SaaS platform that helps small restaurants "
               "reduce food waste by predicting daily demand. "
               "Target customers are independent restaurant owners in India.",
    "country": "India",
}


class TestHealthEndpoint(unittest.TestCase):

    def test_health_returns_200(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})


class TestValidRequest(unittest.TestCase):
    """Test 1: A well-formed request must return 200 and a complete report."""

    @classmethod
    def setUpClass(cls):
        cls.response = client.post("/api/validate", json=VALID_PAYLOAD)
        cls.body     = cls.response.json()

    def test_status_200(self):
        self.assertEqual(self.response.status_code, 200)

    def test_has_all_top_level_keys(self):
        expected = {
            "submission", "ml_result", "idea_evaluation",
            "narrative", "limitations", "scores_are_separate", "combined_score",
        }
        self.assertEqual(set(self.body.keys()), expected)

    def test_submission_echoes_input(self):
        self.assertIn("restaurant", self.body["submission"]["idea_text"])
        self.assertEqual(self.body["submission"]["country_input"], "India")

    def test_ml_result_has_required_fields(self):
        ml = self.body["ml_result"]
        self.assertIn("predicted_class",  ml)
        self.assertIn("pattern_label",    ml)
        self.assertIn("ml_success_score", ml)
        self.assertIn("ml_failure_score", ml)
        self.assertIn("note",             ml)

    def test_ml_predicted_class_is_0_or_1(self):
        self.assertIn(self.body["ml_result"]["predicted_class"], [0, 1])

    def test_ml_scores_sum_to_one(self):
        ml = self.body["ml_result"]
        total = ml["ml_success_score"] + ml["ml_failure_score"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_ml_success_score_in_range(self):
        s = self.body["ml_result"]["ml_success_score"]
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_idea_evaluation_has_required_fields(self):
        ev = self.body["idea_evaluation"]
        self.assertIn("idea_evaluation_score", ev)
        self.assertIn("grade",                 ev)
        self.assertIn("dimensions",            ev)
        self.assertIn("note",                  ev)

    def test_idea_evaluation_score_in_range(self):
        s = self.body["idea_evaluation"]["idea_evaluation_score"]
        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 100)

    def test_five_dimensions_returned(self):
        self.assertEqual(len(self.body["idea_evaluation"]["dimensions"]), 5)

    def test_each_dimension_has_required_fields(self):
        for dim in self.body["idea_evaluation"]["dimensions"]:
            self.assertIn("name",        dim)
            self.assertIn("score",       dim)
            self.assertIn("explanation", dim)

    def test_narrative_has_required_keys(self):
        narr = self.body["narrative"]
        for key in ("strengths", "missing_or_unclear", "guidance"):
            self.assertIn(key, narr)

    def test_limitations_non_empty(self):
        self.assertGreater(len(self.body["limitations"]), 0)

    def test_response_is_json_serialisable(self):
        import json
        json.dumps(self.body)  # must not raise


class TestScoreSeparation(unittest.TestCase):
    """Test 8: ML score and Idea Evaluation Score must remain separate."""

    @classmethod
    def setUpClass(cls):
        cls.body = client.post("/api/validate", json=VALID_PAYLOAD).json()

    def test_combined_score_is_null(self):
        self.assertIsNone(self.body["combined_score"])

    def test_scores_are_separate_flag_true(self):
        self.assertTrue(self.body["scores_are_separate"])

    def test_ml_and_idea_scores_are_different_keys(self):
        self.assertIn("ml_success_score",        self.body["ml_result"])
        self.assertIn("idea_evaluation_score",   self.body["idea_evaluation"])
        # They must not be at the same top-level key
        self.assertNotIn("ml_success_score",     self.body["idea_evaluation"])
        self.assertNotIn("idea_evaluation_score", self.body["ml_result"])


class TestMlScoreNotProbability(unittest.TestCase):
    """Test 9: The ML note must not call the score a probability of success."""

    @classmethod
    def setUpClass(cls):
        cls.body = client.post("/api/validate", json=VALID_PAYLOAD).json()

    def test_ml_note_says_not_probability(self):
        note = self.body["ml_result"]["note"].lower()
        self.assertIn("not", note)

    def test_ml_note_does_not_guarantee_success(self):
        note = self.body["ml_result"]["note"].lower()
        forbidden = ["guaranteed", "will succeed", "certain", "100%"]
        for phrase in forbidden:
            self.assertNotIn(phrase, note,
                f"ML note must not contain '{phrase}'")


class TestEmptyIdea(unittest.TestCase):
    """Test 2 & 3: Empty or whitespace-only idea must return HTTP 400."""

    def test_empty_string_returns_400(self):
        r = client.post("/api/validate", json={"idea": "", "country": "India"})
        self.assertEqual(r.status_code, 400)

    def test_whitespace_only_returns_400(self):
        r = client.post("/api/validate", json={"idea": "   ", "country": "India"})
        self.assertEqual(r.status_code, 400)

    def test_error_response_has_detail_key(self):
        r = client.post("/api/validate", json={"idea": "", "country": "India"})
        self.assertIn("detail", r.json())

    def test_error_detail_does_not_expose_stack_trace(self):
        r = client.post("/api/validate", json={"idea": "", "country": "India"})
        detail = str(r.json().get("detail", ""))
        self.assertNotIn("Traceback", detail)
        self.assertNotIn("File \"", detail)


class TestMissingFields(unittest.TestCase):
    """Tests 4 & 5: Missing required fields must return HTTP 422."""

    def test_missing_idea_returns_422(self):
        r = client.post("/api/validate", json={"country": "India"})
        self.assertEqual(r.status_code, 422)

    def test_missing_country_returns_422(self):
        r = client.post("/api/validate",
                        json={"idea": "An AI platform for restaurants."})
        self.assertEqual(r.status_code, 422)

    def test_empty_body_returns_422(self):
        r = client.post("/api/validate", json={})
        self.assertEqual(r.status_code, 422)


class TestEmptyCountry(unittest.TestCase):
    """Test 6: Empty country string must return HTTP 400."""

    def test_empty_country_returns_400(self):
        r = client.post("/api/validate",
                        json={"idea": "An app for students.", "country": ""})
        self.assertEqual(r.status_code, 400)

    def test_whitespace_country_returns_400(self):
        r = client.post("/api/validate",
                        json={"idea": "An app for students.", "country": "  "})
        self.assertEqual(r.status_code, 400)


class TestUnresolvableCountry(unittest.TestCase):
    """
    Test 7: A country string that cannot be resolved to an ISO-3 code
    must still return 200 — the pipeline handles this gracefully by
    setting country_other=1 and flagging it in submission_warnings.
    """

    def test_unrecognised_country_returns_200(self):
        r = client.post("/api/validate", json={
            "idea":    "An e-commerce platform for local artisans.",
            "country": "Xanadu",        # fictional, unresolvable
        })
        self.assertEqual(r.status_code, 200)

    def test_unrecognised_country_reflected_in_response(self):
        r = client.post("/api/validate", json={
            "idea":    "An e-commerce platform for local artisans.",
            "country": "Xanadu",
        })
        body = r.json()
        # The pipeline should not crash; country_input should echo back
        self.assertEqual(body["submission"]["country_input"], "Xanadu")
        # country_other flag should be active
        self.assertIn(
            "country_other",
            body["ml_result"]["active_country_features"],
        )


class TestPipelineFailure(unittest.TestCase):
    """
    Test 10: Simulate a pipeline failure to confirm the API returns
    HTTP 500 with a safe generic message and no stack trace.
    """

    def test_pipeline_failure_returns_500(self):
        with patch("api.app.generate_report", side_effect=RuntimeError("disk full")):
            r = client.post("/api/validate", json=VALID_PAYLOAD)
        self.assertEqual(r.status_code, 500)

    def test_500_response_has_detail_key(self):
        with patch("api.app.generate_report", side_effect=RuntimeError("disk full")):
            r = client.post("/api/validate", json=VALID_PAYLOAD)
        self.assertIn("detail", r.json())

    def test_500_does_not_expose_stack_trace(self):
        with patch("api.app.generate_report", side_effect=RuntimeError("disk full")):
            r = client.post("/api/validate", json=VALID_PAYLOAD)
        detail = str(r.json().get("detail", ""))
        self.assertNotIn("Traceback", detail)
        self.assertNotIn("disk full", detail)   # internal message not leaked
        self.assertNotIn("RuntimeError", detail)


# ---------------------------------------------------------------------------
# STANDALONE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestHealthEndpoint,
        TestValidRequest,
        TestScoreSeparation,
        TestMlScoreNotProbability,
        TestEmptyIdea,
        TestMissingFields,
        TestEmptyCountry,
        TestUnresolvableCountry,
        TestPipelineFailure,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
