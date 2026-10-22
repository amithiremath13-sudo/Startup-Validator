"""
tests/test_idea_scorer.py
==========================
Phase 4B — Idea Evaluation Scoring Engine Tests

Tests the Component B scoring engine (inference/idea_scorer.py)
in isolation and integrated with Component A via run_full_evaluation().

TEST IDEAS
----------
1. Specific, well-described startup (high expected score)
2. Clear education platform
3. Healthcare idea with problem + solution
4. Invoice/finance software with explicit categories
5. Deliberately vague ("I have an app idea") — low expected score
6. Empty string — graceful handling
7. Structural contract tests (score ranges, weight sums, separation)

HOW TO RUN
----------
    python -m pytest tests/test_idea_scorer.py -v
    python tests/test_idea_scorer.py
"""

import os
import sys
import unittest

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.idea_scorer import (
    DIMENSION_WEIGHTS,
    MAX_DIMENSION_SCORE,
    MAX_OVERALL_SCORE,
    DimensionResult,
    IdeaScorerResult,
    score_idea,
)
from inference.feature_extractor import StartupIdeaInput

# 05_evaluate_idea.py starts with a digit so we import via importlib
import importlib.util, os as _os
_spec = importlib.util.spec_from_file_location(
    "evaluate_idea",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                  "05_evaluate_idea.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_full_evaluation = _mod.run_full_evaluation


class TestIdeaScorerUnit(unittest.TestCase):
    """Unit tests for score_idea() in isolation — no ML model loaded."""

    # ------------------------------------------------------------------
    # STRUCTURAL CONTRACT TESTS
    # ------------------------------------------------------------------

    def test_weights_sum_to_one(self):
        """DIMENSION_WEIGHTS must sum to exactly 1.0."""
        self.assertAlmostEqual(sum(DIMENSION_WEIGHTS.values()), 1.0, places=9)

    def test_returns_correct_type(self):
        result = score_idea("A platform that helps farmers track crops using AI.")
        self.assertIsInstance(result, IdeaScorerResult)

    def test_returns_five_dimensions(self):
        result = score_idea("An e-commerce platform for small businesses in India.")
        self.assertEqual(len(result.dimensions), 5)

    def test_all_dimension_keys_present(self):
        result = score_idea("A healthcare app for remote patient monitoring.")
        keys = {d.key for d in result.dimensions}
        expected = set(DIMENSION_WEIGHTS.keys())
        self.assertEqual(keys, expected)

    def test_dimension_scores_in_range(self):
        """Every dimension score must be an int in [0, MAX_DIMENSION_SCORE]."""
        result = score_idea("An AI-powered invoicing tool for freelancers.")
        for d in result.dimensions:
            self.assertIsInstance(d.score, int,
                f"Dimension '{d.name}' score is not int: {d.score!r}")
            self.assertGreaterEqual(d.score, 0,
                f"Dimension '{d.name}' score below 0")
            self.assertLessEqual(d.score, MAX_DIMENSION_SCORE,
                f"Dimension '{d.name}' score above {MAX_DIMENSION_SCORE}")

    def test_overall_score_in_range(self):
        """Overall score must be an int in [0, MAX_OVERALL_SCORE]."""
        for text in [
            "",
            "app",
            "We build an AI platform helping restaurants reduce food waste in India.",
            "A software tool that helps small businesses manage invoices automatically.",
        ]:
            result = score_idea(text)
            self.assertIsInstance(result.idea_evaluation_score, int,
                f"Overall score not int for: {text!r}")
            self.assertGreaterEqual(result.idea_evaluation_score, 0)
            self.assertLessEqual(result.idea_evaluation_score, MAX_OVERALL_SCORE)

    def test_weighted_scores_sum_to_overall(self):
        """Sum of weighted_score values must equal (approx) idea_evaluation_score."""
        result = score_idea("A fintech platform connecting lenders and borrowers.")
        computed = round(sum(d.weighted_score for d in result.dimensions))
        self.assertEqual(computed, result.idea_evaluation_score,
            f"Weighted sum {computed} != overall {result.idea_evaluation_score}")

    def test_caveats_always_present(self):
        """There must always be at least one caveat in the result."""
        result = score_idea("We build software.")
        self.assertGreater(len(result.caveats), 0)

    def test_caveats_contain_no_ml_success_claim(self):
        """
        Caveats must not claim to predict startup success probability.
        The phrase 'probability of success' (as a factual claim) must
        not appear in the caveat text.
        """
        result = score_idea("A healthcare tech platform.")
        for caveat in result.caveats:
            self.assertNotIn(
                "probability of success", caveat.lower(),
                "Caveat must not claim to predict probability of success"
            )

    def test_dimension_explanations_are_non_empty(self):
        result = score_idea("A marketplace for buying and selling second-hand electronics.")
        for d in result.dimensions:
            self.assertIsInstance(d.explanation, str)
            self.assertGreater(len(d.explanation.strip()), 0,
                f"Dimension '{d.name}' has empty explanation")

    def test_word_count_is_correct(self):
        text   = "This is a test with exactly eight words here."
        result = score_idea(text)
        self.assertEqual(result.word_count, 9)  # 9 words in that sentence

    def test_word_count_zero_for_empty(self):
        result = score_idea("")
        self.assertEqual(result.word_count, 0)

    # ------------------------------------------------------------------
    # TEST IDEA 1 — Specific, well-described startup
    # ------------------------------------------------------------------

    def test_idea1_specific_well_described(self):
        """
        A detailed, specific idea should score noticeably higher than a
        vague idea across all 5 dimensions.
        """
        text = (
            "We are building an AI-powered platform that helps small restaurant "
            "owners reduce food waste by predicting daily demand using historical "
            "sales data and weather patterns. Our SaaS tool integrates with "
            "existing POS systems and provides a real-time dashboard for "
            "inventory management. Target customers are independent restaurant "
            "owners in India struggling with over-ordering and spoilage costs."
        )
        result = score_idea(text)

        # Should score meaningfully above the midpoint
        self.assertGreaterEqual(result.idea_evaluation_score, 55,
            f"Specific idea should score >= 55, got {result.idea_evaluation_score}")

        # All dimensions should have non-zero scores
        for d in result.dimensions:
            self.assertGreater(d.score, 0,
                f"Dimension '{d.name}' should be non-zero for a specific idea")

        # Submission should be flagged meaningful
        self.assertTrue(result.is_meaningful_submission)

        print(f"\n  TEST_1 specific idea score: {result.idea_evaluation_score}/100")
        for d in result.dimensions:
            print(f"    {d.name:<22}: {d.score}/10")

    # ------------------------------------------------------------------
    # TEST IDEA 2 — Education platform
    # ------------------------------------------------------------------

    def test_idea2_education_platform(self):
        """
        A clear education platform idea — should have good solution and
        target market scores.
        """
        text = (
            "Online education platform for students preparing for competitive "
            "exams in India. The platform provides video lessons, practice tests, "
            "and personalized study plans. Students struggle to find quality "
            "affordable coaching for government job examinations."
        )
        result = score_idea(text)

        self.assertGreaterEqual(result.idea_evaluation_score, 45)
        self.assertTrue(result.is_meaningful_submission)

        # Solution and target market should be clear
        sol = next(d for d in result.dimensions if d.key == "solution_clarity")
        mkt = next(d for d in result.dimensions if d.key == "target_market")
        self.assertGreater(sol.score, 2,
            f"Solution clarity should be > 2 for education platform, got {sol.score}")
        self.assertGreater(mkt.score, 2,
            f"Target market should be > 2 (students, India), got {mkt.score}")

        print(f"\n  TEST_2 education platform score: {result.idea_evaluation_score}/100")

    # ------------------------------------------------------------------
    # TEST IDEA 3 — Healthcare remote monitoring
    # ------------------------------------------------------------------

    def test_idea3_healthcare_remote_monitoring(self):
        """
        Healthcare technology platform for remote patient monitoring.
        Should score moderate — has solution language and market signals.
        """
        text = (
            "Healthcare technology platform for remote patient monitoring. "
            "Doctors need a way to track patients with chronic conditions "
            "between hospital visits. Our platform uses IoT sensors and a "
            "mobile app to collect vital signs and alert clinicians to "
            "abnormal readings in real time."
        )
        result = score_idea(text)

        self.assertGreaterEqual(result.idea_evaluation_score, 40)
        self.assertTrue(result.is_meaningful_submission)

        # Problem clarity should be non-zero (doctors need a way, hospital visits)
        prob = next(d for d in result.dimensions if d.key == "problem_clarity")
        self.assertGreater(prob.score, 0,
            f"Problem clarity should be > 0 for healthcare monitoring, got {prob.score}")

        print(f"\n  TEST_3 healthcare score: {result.idea_evaluation_score}/100")

    # ------------------------------------------------------------------
    # TEST IDEA 4 — Invoice software (software + finance)
    # ------------------------------------------------------------------

    def test_idea4_invoice_software(self):
        """
        A software platform for small businesses to manage invoices.
        Clear solution (platform, software) and target (small businesses).
        """
        text = (
            "A software platform for small businesses to manage invoices. "
            "Small business owners spend hours creating and tracking invoices "
            "manually in spreadsheets. Our SaaS tool automates invoice generation, "
            "payment reminders, and bookkeeping. Target customers are freelancers "
            "and small business owners in the USA."
        )
        result = score_idea(text)

        self.assertGreaterEqual(result.idea_evaluation_score, 50)
        self.assertTrue(result.is_meaningful_submission)

        # Both problem and solution should be clear
        prob = next(d for d in result.dimensions if d.key == "problem_clarity")
        sol  = next(d for d in result.dimensions if d.key == "solution_clarity")
        self.assertGreater(prob.score, 2)
        self.assertGreater(sol.score, 2)

        print(f"\n  TEST_4 invoice software score: {result.idea_evaluation_score}/100")

    # ------------------------------------------------------------------
    # TEST IDEA 5 — Deliberately vague
    # ------------------------------------------------------------------

    def test_idea5_vague_idea(self):
        """
        'I have an app idea.' — very vague, should score LOW.
        Ensures the engine does not award high scores to thin submissions.
        """
        result = score_idea("I have an app idea.")

        # Must score below 30 — there is almost no information here
        self.assertLess(result.idea_evaluation_score, 30,
            f"Vague idea should score < 30, got {result.idea_evaluation_score}")

        # Specificity should be very low
        spec = next(d for d in result.dimensions if d.key == "specificity")
        self.assertLessEqual(spec.score, 4,
            f"Specificity should be <= 4 for 'I have an app idea', got {spec.score}")

        # Should still be meaningful (5 words = exactly the threshold)
        # "I have an app idea" is 5 words — is_meaningful may be True or False
        # but the pipeline must not crash
        self.assertIsNotNone(result.is_meaningful_submission)

        print(f"\n  TEST_5 vague idea score: {result.idea_evaluation_score}/100")
        for d in result.dimensions:
            print(f"    {d.name:<22}: {d.score}/10")

    # ------------------------------------------------------------------
    # TEST IDEA 6 — Empty string
    # ------------------------------------------------------------------

    def test_idea6_empty_string(self):
        """
        Empty string must not crash. All scores should be 0.
        is_meaningful_submission must be False.
        """
        result = score_idea("")

        self.assertEqual(result.idea_evaluation_score, 0)
        self.assertEqual(result.word_count, 0)
        self.assertFalse(result.is_meaningful_submission)
        for d in result.dimensions:
            self.assertEqual(d.score, 0,
                f"All dimension scores should be 0 for empty input, "
                f"got {d.name}={d.score}")

        print(f"\n  TEST_6 empty string score: {result.idea_evaluation_score}/100 (expected 0)")

    # ------------------------------------------------------------------
    # MONOTONICITY CHECK — specific ideas must outscore vague ones
    # ------------------------------------------------------------------

    def test_specific_beats_vague(self):
        """
        A detailed, specific idea must score higher than a vague one.
        This is the fundamental monotonicity property of the scorer.
        """
        vague = score_idea("I have an app idea.")
        specific = score_idea(
            "We build a B2B SaaS platform that automates invoice generation "
            "for small businesses. Our customers struggle with manual "
            "bookkeeping and late payments. The platform integrates with "
            "accounting software and sends automated payment reminders."
        )
        self.assertGreater(
            specific.idea_evaluation_score,
            vague.idea_evaluation_score,
            f"Specific idea ({specific.idea_evaluation_score}) must beat "
            f"vague idea ({vague.idea_evaluation_score})"
        )

    def test_more_complete_beats_less_complete(self):
        """Adding more relevant detail to an idea should increase the score."""
        brief = score_idea("An app for students.")
        detailed = score_idea(
            "An online learning platform for students preparing for competitive "
            "entrance exams. Students struggle to access quality study materials "
            "affordably. Our platform provides video lessons, mock tests, and "
            "personalised progress tracking. Target market: engineering aspirants "
            "in India."
        )
        self.assertGreater(detailed.idea_evaluation_score, brief.idea_evaluation_score)


class TestFullEvaluation(unittest.TestCase):
    """
    Integration tests: run_full_evaluation() (Component A + Component B).
    These load the trained model, so they are slightly slower.
    """

    @classmethod
    def setUpClass(cls):
        """Load model once for all integration tests."""
        from inference.model_loader import load_model_and_metadata
        cls.model, cls.feature_names, cls.prep_meta = load_model_and_metadata()

    def _evaluate(self, idea_text, country=None, categories=None):
        inp = StartupIdeaInput(
            idea_text  = idea_text,
            country    = country,
            categories = categories or [],
        )
        return run_full_evaluation(inp)

    def test_full_eval_returns_both_components(self):
        """run_full_evaluation must return scorer_result and ML outputs."""
        out = self._evaluate(
            "A fintech platform for digital payments in India.",
            country="India"
        )
        self.assertIn("scorer_result",      out)
        self.assertIn("extraction_result",  out)
        self.assertIn("predicted_class",    out)
        self.assertIn("prob_success",       out)
        self.assertIn("prob_failure",       out)

    def test_ml_score_and_idea_score_are_independent(self):
        """
        Component A and Component B must produce independent outputs.
        The idea_evaluation_score must not equal the ML prob_success.
        They measure different things and live on different scales.
        """
        out = self._evaluate(
            "We help small restaurants reduce food waste using AI.",
            country="India"
        )
        ml_score   = out["prob_success"]        # float 0.0–1.0
        idea_score = out["scorer_result"].idea_evaluation_score  # int 0–100

        # They are on different scales — confirm they are not accidentally equal
        # (idea_score/100 vs ml_score)
        self.assertNotAlmostEqual(
            idea_score / 100, ml_score, places=2,
            msg="Idea score and ML score should not be numerically equal "
                "(they measure different things)"
        )

    def test_full_eval_predicted_class_is_0_or_1(self):
        out = self._evaluate("An education app for students.", country="India")
        self.assertIn(out["predicted_class"], [0, 1])

    def test_full_eval_probabilities_sum_to_one(self):
        out = self._evaluate(
            "A software tool for invoice management.",
            country="USA", categories=["Software", "Finance"]
        )
        total = out["prob_success"] + out["prob_failure"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_full_eval_feature_count_unchanged(self):
        """Component B must not alter the 65-feature vector."""
        out = self._evaluate(
            "A healthcare monitoring platform.",
            country="USA"
        )
        fv = out["extraction_result"].feature_vector
        self.assertEqual(len(fv), 65)

    def test_full_eval_vague_idea_still_runs(self):
        """Even a vague idea must not crash the full evaluation."""
        out = self._evaluate("I have an app idea.")
        self.assertIsNotNone(out["scorer_result"])
        self.assertIn(out["predicted_class"], [0, 1])
        # Vague idea should score low on Component B
        self.assertLess(out["scorer_result"].idea_evaluation_score, 30)

    def test_scores_not_combined(self):
        """
        The result dict must NOT contain a field called 'combined_score'
        or 'final_score' — the two scores must remain separate until
        explicitly designed for combination.
        """
        out = self._evaluate("A marketplace for buying secondhand goods.", country="USA")
        self.assertNotIn("combined_score", out)
        self.assertNotIn("final_score",    out)
        self.assertNotIn("combined",       out)


class TestSubstantivenessValidation(unittest.TestCase):
    """
    Regression tests for the substantiveness guard added to score_idea().

    is_substantive_submission is a content-signal check (>= 2 keyword hits
    across all four content lists) that is separate from is_meaningful_submission
    (the word-count gate, >= 5 words).

    Vague filler phrases must set is_substantive_submission = False and
    include a warning in caveats.  Specific short ideas must still be accepted.
    Empty / whitespace input is unchanged: is_meaningful_submission = False.
    """

    def test_vague_app_idea_not_substantive(self):
        result = score_idea("I have an app idea.")
        self.assertFalse(result.is_substantive_submission,
            "Expected is_substantive_submission=False for 'I have an app idea.'")
        warning_text = " ".join(result.caveats).lower()
        self.assertIn("problem", warning_text,
            "Warning should mention 'problem' to guide the user")

    def test_vague_start_business_not_substantive(self):
        result = score_idea("I want to start a business.")
        self.assertFalse(result.is_substantive_submission,
            "Expected is_substantive_submission=False for 'I want to start a business.'")

    def test_vague_startup_not_substantive(self):
        result = score_idea("My idea is a startup.")
        self.assertFalse(result.is_substantive_submission,
            "Expected is_substantive_submission=False for 'My idea is a startup.'")

    def test_specific_short_idea_is_substantive(self):
        """'An app that reduces waste.' -- short but contains content signals."""
        result = score_idea("An app that reduces waste.")
        self.assertTrue(result.is_substantive_submission,
            "Expected is_substantive_submission=True for 'An app that reduces waste.'")
        self.assertTrue(result.is_meaningful_submission)

    def test_empty_string_not_meaningful_or_substantive(self):
        result = score_idea("")
        self.assertFalse(result.is_meaningful_submission)
        self.assertFalse(result.is_substantive_submission)

    def test_whitespace_not_meaningful_or_substantive(self):
        result = score_idea("   ")
        self.assertFalse(result.is_meaningful_submission)
        self.assertFalse(result.is_substantive_submission)

    def test_flags_are_independent(self):
        """Vague idea: is_meaningful=True (word count) but is_substantive=False (content)."""
        result = score_idea("I have an app idea.")
        self.assertTrue(result.is_meaningful_submission,
            "is_meaningful_submission must remain True (5 words >= threshold)")
        self.assertFalse(result.is_substantive_submission,
            "is_substantive_submission must be False (only 1 content signal)")

    def test_specific_idea_both_flags_true(self):
        result = score_idea(
            "An AI-powered SaaS platform helping small restaurants "
            "reduce food waste. Target: independent restaurant owners in India."
        )
        self.assertTrue(result.is_meaningful_submission)
        self.assertTrue(result.is_substantive_submission)


# ---------------------------------------------------------------------------
# STANDALONE EXECUTION
# ---------------------------------------------------------------------------

def run_all_tests_standalone():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestIdeaScorerUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestFullEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestSubstantivenessValidation))
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests_standalone()
    sys.exit(0 if success else 1)
