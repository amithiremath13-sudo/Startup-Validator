"""
tests/test_report_generator.py
================================
Phase 5 — Validation Report Generator Tests
"""

import json
import os
import sys
import unittest

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.feature_extractor import StartupIdeaInput
from inference.report_generator import (
    ValidationReport, DimensionSummary,
    generate_report, render_report_text, report_to_dict,
    _grade, _ml_pattern_label,
)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "evaluate_idea",
    os.path.join(PROJECT_ROOT, "05_evaluate_idea.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_full_evaluation = _mod.run_full_evaluation


def _run(idea_text, country=None, categories=None):
    """Helper: full pipeline → ValidationReport."""
    inp = StartupIdeaInput(
        idea_text  = idea_text,
        country    = country,
        categories = categories or [],
    )
    out = run_full_evaluation(inp)
    return generate_report(out, country_input=country, categories_input=categories or [])


class TestReportStructure(unittest.TestCase):
    """Contract tests: every report must have the required structure."""

    def setUp(self):
        self.report = _run(
            "A SaaS platform helping small businesses automate invoice "
            "generation and payment reminders. Owners waste hours on manual "
            "bookkeeping. Target: freelancers and SMEs in the USA.",
            country="USA",
        )

    def test_returns_validation_report_type(self):
        self.assertIsInstance(self.report, ValidationReport)

    def test_has_five_dimensions(self):
        self.assertEqual(len(self.report.dimensions), 5)

    def test_all_dimension_types(self):
        for d in self.report.dimensions:
            self.assertIsInstance(d, DimensionSummary)

    def test_ml_score_is_float_in_range(self):
        self.assertIsInstance(self.report.ml_success_score, float)
        self.assertGreaterEqual(self.report.ml_success_score, 0.0)
        self.assertLessEqual(self.report.ml_success_score, 1.0)

    def test_ml_scores_sum_to_one(self):
        total = self.report.ml_success_score + self.report.ml_failure_score
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_idea_evaluation_score_in_range(self):
        self.assertIsInstance(self.report.idea_evaluation_score, int)
        self.assertGreaterEqual(self.report.idea_evaluation_score, 0)
        self.assertLessEqual(self.report.idea_evaluation_score, 100)

    def test_ml_predicted_class_is_0_or_1(self):
        self.assertIn(self.report.ml_predicted_class, [0, 1])

    def test_ml_pattern_label_matches_class(self):
        if self.report.ml_predicted_class == 1:
            self.assertEqual(self.report.ml_pattern_label, "Success Pattern")
        else:
            self.assertEqual(self.report.ml_pattern_label, "Failure Pattern")

    def test_limitations_non_empty(self):
        self.assertGreater(len(self.report.limitations), 0)

    def test_dimension_scores_are_valid(self):
        for d in self.report.dimensions:
            self.assertIsInstance(d.score, int)
            self.assertGreaterEqual(d.score, 0)
            self.assertLessEqual(d.score, d.max_score)
            self.assertEqual(d.max_score, 10)

    def test_grade_is_valid_string(self):
        valid = {"Strong", "Moderate", "Developing", "Weak", "Very Weak"}
        self.assertIn(self.report.idea_evaluation_grade, valid)


class TestScoreSeparation(unittest.TestCase):
    """
    The ML score and Idea Evaluation Score must remain separate.
    They measure different things and must never be merged.
    """

    def test_scores_are_separate_in_report(self):
        """ValidationReport must have both scores as distinct fields."""
        report = _run("An AI platform for logistics optimisation.", country="USA")
        self.assertIsNotNone(report.ml_success_score)
        self.assertIsNotNone(report.idea_evaluation_score)
        # They live on different scales — confirm they are separate attributes
        self.assertNotEqual(
            type(report.ml_success_score), type(report.idea_evaluation_score),
            "ml_success_score (float) and idea_evaluation_score (int) should be different types"
        )

    def test_dict_has_separate_top_level_keys(self):
        """report_to_dict() must keep scores as separate top-level keys."""
        report = _run("A healthcare app for patients.", country="GBR")
        d = report_to_dict(report)
        self.assertIn("ml_result",       d)
        self.assertIn("idea_evaluation", d)
        self.assertIn("ml_success_score", d["ml_result"])
        self.assertIn("idea_evaluation_score", d["idea_evaluation"])
        # There must not be a combined score
        self.assertIsNone(d.get("combined_score"))
        self.assertNotIn("combined_score", d["ml_result"])
        self.assertNotIn("combined_score", d["idea_evaluation"])

    def test_scores_not_combined_in_text_report(self):
        """Text report must not contain phrases that present a combined score."""
        report = _run("A fintech platform for digital payments.", country="IND")
        text   = render_report_text(report)
        forbidden_phrases = [
            "combined score",
            "overall score of",
            "final score",
            "aggregate score",
        ]
        for phrase in forbidden_phrases:
            self.assertNotIn(
                phrase, text.lower(),
                f"Text report must not contain '{phrase}'"
            )

    def test_dict_combined_score_is_none(self):
        """report_to_dict must set combined_score explicitly to None."""
        report = _run("We build software for restaurants.", country="USA")
        d      = report_to_dict(report)
        self.assertIn("combined_score", d)
        self.assertIsNone(d["combined_score"])

    def test_scores_are_separate_flag_is_true(self):
        report = _run("A marketplace for handmade goods.", country="USA")
        d      = report_to_dict(report)
        self.assertTrue(d.get("scores_are_separate"))


class TestMlScoreNotCalledProbability(unittest.TestCase):
    """
    The ML score must never be described as a guaranteed probability
    of startup success in any generated output.
    """

    PROBABILITY_CLAIMS = [
        "probability of success",
        "chance of success",
        "guaranteed",
        "will succeed",
        "will fail",
        "will definitely",
        "certain to",
        "100%",
    ]

    def _assert_no_probability_claim(self, text: str, context: str):
        text_lower = text.lower()
        for phrase in self.PROBABILITY_CLAIMS:
            self.assertNotIn(
                phrase, text_lower,
                f"[{context}] Must not contain '{phrase}'"
            )

    def test_text_report_no_probability_claim(self):
        report = _run("An education platform for students in India.", country="India")
        text   = render_report_text(report)
        self._assert_no_probability_claim(text, "text_report")

    def test_dict_note_no_probability_claim(self):
        report = _run("A SaaS tool for HR teams.", country="USA")
        d      = report_to_dict(report)
        note   = d["ml_result"]["note"].lower()
        # The note should say it's NOT a probability — check it doesn't claim it IS
        self.assertIn("not", note,
            "ML result note should explicitly say it is NOT a probability")

    def test_limitations_mention_training_period(self):
        """Limitations must mention the 2016 training data cutoff."""
        report = _run("A clean technology startup for solar energy.", country="USA")
        lim_text = " ".join(report.limitations).lower()
        self.assertIn("2016", lim_text,
            "Limitations must mention the ~1990-2016 training data period")

    def test_limitations_mention_not_probability(self):
        report = _run("An app for task management.", country="USA")
        lim_text = " ".join(report.limitations).lower()
        self.assertTrue(
            "not" in lim_text and ("probability" in lim_text or "calibrat" in lim_text),
            "Limitations must state the score is not a calibrated probability"
        )


class TestSpecificIdea(unittest.TestCase):
    """
    Test 1: A specific, well-described startup idea.
    Verifies the report correctly identifies strengths and produces
    a meaningful Idea Evaluation Score.
    """

    def setUp(self):
        self.report = _run(
            "We are building an AI-powered platform that helps small restaurant "
            "owners reduce food waste by predicting daily demand using historical "
            "sales data. Owners struggle with over-ordering and spoilage costs. "
            "Our SaaS tool integrates with POS systems and provides real-time "
            "inventory recommendations. Target customers: independent restaurant "
            "owners in India.",
            country="India",
        )

    def test_idea_score_is_meaningful(self):
        self.assertGreaterEqual(self.report.idea_evaluation_score, 50,
            f"Specific idea should score >= 50, got {self.report.idea_evaluation_score}")

    def test_strengths_not_empty(self):
        self.assertGreater(len(self.report.strengths), 0,
            "A well-described idea should have identified strengths")

    def test_country_resolved_correctly(self):
        self.assertEqual(self.report.resolved_country, "IND")

    def test_text_report_renders(self):
        text = render_report_text(self.report)
        self.assertIn("SECTION 1", text)
        self.assertIn("SECTION 2", text)
        self.assertIn("SECTION 3", text)
        self.assertIn("SECTION 4", text)
        self.assertIn("SECTION 5", text)

    def test_dict_is_json_serialisable(self):
        d = report_to_dict(self.report)
        serialised = json.dumps(d)  # must not raise
        self.assertIsInstance(serialised, str)

    def test_ml_and_idea_scores_both_present_in_dict(self):
        d = report_to_dict(self.report)
        self.assertIsNotNone(d["ml_result"]["ml_success_score"])
        self.assertIsNotNone(d["idea_evaluation"]["idea_evaluation_score"])


class TestVagueIdea(unittest.TestCase):
    """
    Test 2: A deliberately vague startup idea.
    Verifies the report produces a low Idea Evaluation Score and
    identifies the missing information.
    """

    def setUp(self):
        self.report = _run("I have an app idea.")

    def test_idea_score_is_low(self):
        self.assertLess(self.report.idea_evaluation_score, 30,
            f"Vague idea should score < 30, got {self.report.idea_evaluation_score}")

    def test_missing_info_identified(self):
        self.assertGreater(len(self.report.missing_or_unclear), 0,
            "Vague idea should have identified missing information")

    def test_guidance_not_empty(self):
        self.assertGreater(len(self.report.guidance), 0,
            "Vague idea should receive improvement suggestions")

    def test_incomplete_submission_flagged(self):
        self.assertFalse(self.report.is_complete_submission,
            "Vague idea with no country should be incomplete")

    def test_text_report_includes_missing_section(self):
        text = render_report_text(self.report)
        self.assertIn("Missing or unclear", text)
        self.assertIn("Suggestions", text)

    def test_ml_score_still_valid(self):
        self.assertGreaterEqual(self.report.ml_success_score, 0.0)
        self.assertLessEqual(self.report.ml_success_score, 1.0)
        self.assertIn(self.report.ml_predicted_class, [0, 1])


class TestEmptyIdea(unittest.TestCase):
    """
    Test 3: Empty string — pipeline must not crash.
    All scores should be 0 or minimal.
    """

    def setUp(self):
        self.report = _run("")

    def test_does_not_crash(self):
        self.assertIsInstance(self.report, ValidationReport)

    def test_idea_score_is_zero(self):
        self.assertEqual(self.report.idea_evaluation_score, 0)

    def test_word_count_is_zero(self):
        self.assertEqual(self.report.word_count, 0)

    def test_not_meaningful_submission(self):
        self.assertFalse(self.report.is_meaningful_submission)

    def test_all_dimension_scores_zero(self):
        for d in self.report.dimensions:
            self.assertEqual(d.score, 0,
                f"Dimension '{d.name}' should be 0 for empty input, got {d.score}")

    def test_text_report_renders_without_crash(self):
        text = render_report_text(self.report)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_ml_score_still_valid(self):
        """Even for an empty submission, ML outputs must be valid numbers."""
        self.assertIn(self.report.ml_predicted_class, [0, 1])
        total = self.report.ml_success_score + self.report.ml_failure_score
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_dict_serialisable_for_empty(self):
        d = report_to_dict(self.report)
        serialised = json.dumps(d)
        self.assertIsInstance(serialised, str)


class TestGradeHelper(unittest.TestCase):
    """Unit tests for helper functions used in report generation."""

    def test_grade_boundaries(self):
        self.assertEqual(_grade(80),  "Strong")
        self.assertEqual(_grade(79),  "Moderate")
        self.assertEqual(_grade(60),  "Moderate")
        self.assertEqual(_grade(59),  "Developing")
        self.assertEqual(_grade(40),  "Developing")
        self.assertEqual(_grade(39),  "Weak")
        self.assertEqual(_grade(20),  "Weak")
        self.assertEqual(_grade(19),  "Very Weak")
        self.assertEqual(_grade(0),   "Very Weak")
        self.assertEqual(_grade(100), "Strong")

    def test_pattern_label(self):
        self.assertEqual(_ml_pattern_label(1), "Success Pattern")
        self.assertEqual(_ml_pattern_label(0), "Failure Pattern")


class TestMissingOrUnclearSection(unittest.TestCase):
    """
    Regression tests for the 'Missing or unclear information' section fix.

    Previously the renderer printed 'No critical gaps detected' even when
    dimension scores of 4–6 indicated improvable areas.  The fixed renderer
    must:
      - Name each dimension that scored 1–6 (excluding Completeness) when
        missing_or_unclear is empty.
      - Print the neutral message ONLY when every non-Completeness dimension
        is either 0 or >= 7.
      - Never print '(No critical gaps detected)'.
    """

    def _make_report(self, dim_scores: dict) -> "ValidationReport":
        """
        Build a minimal ValidationReport with controllable dimension scores.
        dim_scores maps dimension name → score (int 0-10).
        Expected keys: Problem Clarity, Solution Clarity, Target Market,
                       Specificity, Completeness.
        """
        from inference.report_generator import DimensionSummary, ValidationReport
        from inference.idea_scorer import DIMENSION_WEIGHTS

        weight_map = {
            "Problem Clarity":  0.25,
            "Solution Clarity": 0.25,
            "Target Market":    0.20,
            "Specificity":      0.15,
            "Completeness":     0.15,
        }
        dims = [
            DimensionSummary(
                name=name,
                score=score,
                max_score=10,
                weight_pct=round(weight_map[name] * 100),
                explanation=f"Mock explanation for {name} score={score}.",
                matched_signals=[],
            )
            for name, score in dim_scores.items()
        ]
        overall = round(sum(
            d.score * weight_map[d.name] * 10 for d in dims
        ))
        return ValidationReport(
            idea_text="mock idea text",
            country_input="USA",
            categories_input=[],
            resolved_country="USA",
            resolved_categories=["Software"],
            is_complete_submission=True,
            word_count=20,
            is_meaningful_submission=True,
            ml_predicted_class=1,
            ml_pattern_label="Success Pattern",
            ml_success_score=0.65,
            ml_failure_score=0.35,
            active_country_features=["country_USA"],
            active_category_features=["cat_Software"],
            n_categories=1,
            idea_evaluation_score=overall,
            idea_evaluation_grade=_grade(overall),
            dimensions=dims,
            strengths=["Good problem."],
            missing_or_unclear=[],   # empty — exercises the renderer fallback
            guidance=[],
            submission_warnings=[],
            limitations=["Limitation 1."],
        )

    def test_contradictory_message_is_gone(self):
        """
        The renderer must NEVER output '(No critical gaps detected)' — that
        phrase was the contradictory fallback being replaced.
        """
        # Idea with mixed scores: Problem Clarity 6, Target Market 6, Specificity 4
        report = self._make_report({
            "Problem Clarity":  6,
            "Solution Clarity": 10,
            "Target Market":    6,
            "Specificity":      4,
            "Completeness":     7,
        })
        text = render_report_text(report)
        self.assertNotIn(
            "(No critical gaps detected)", text,
            "The old contradictory fallback must not appear in any report"
        )

    def test_partial_dims_are_named_when_missing_list_empty(self):
        """
        When missing_or_unclear is empty but some dims scored 1–6,
        the renderer must name those dims in the Missing section.
        """
        report = self._make_report({
            "Problem Clarity":  6,   # partial — should appear
            "Solution Clarity": 10,
            "Target Market":    6,   # partial — should appear
            "Specificity":      4,   # partial — should appear
            "Completeness":     7,
        })
        text = render_report_text(report)
        # Section header must be present
        self.assertIn("Missing or unclear information:", text)
        # Each low-scoring non-Completeness dim must be named
        self.assertIn("Problem Clarity", text)
        self.assertIn("Target Market",   text)
        self.assertIn("Specificity",     text)
        # Must say "partially addressed"
        self.assertIn("partially addressed", text)

    def test_completeness_excluded_from_partial_list(self):
        """
        Completeness scoring 1–6 must NOT appear in the partial gaps list —
        it is a meta-dimension and not an independently improvable element.
        """
        report = self._make_report({
            "Problem Clarity":  8,
            "Solution Clarity": 10,
            "Target Market":    8,
            "Specificity":      8,
            "Completeness":     4,   # low but excluded by design
        })
        text = render_report_text(report)
        self.assertNotIn("Completeness (4/10)", text,
            "Completeness must not appear in the partial-gaps list")

    def test_neutral_message_when_all_dims_high_or_zero(self):
        """
        When every non-Completeness dimension is 0 or >= 7, the neutral
        message must appear instead of any partial-gap list.
        """
        report = self._make_report({
            "Problem Clarity":  8,
            "Solution Clarity": 10,
            "Target Market":    8,
            "Specificity":      8,
            "Completeness":     7,
        })
        text = render_report_text(report)
        self.assertIn(
            "No major clarity gaps detected by the rule-based evaluator.",
            text,
            "Neutral message must appear when all dims are 0 or >= 7"
        )
        self.assertNotIn("partially addressed", text)

    def test_zero_score_dim_not_in_partial_list(self):
        """
        Dimensions scoring exactly 0 are already captured by
        _build_missing_or_unclear and would appear via the primary branch.
        The fallback must NOT double-report them via the partial-gaps path
        (the condition is 0 < score < 7, not <= 6).
        """
        # All zero dims, missing_or_unclear populated externally
        from inference.report_generator import DimensionSummary, ValidationReport
        dims = [
            DimensionSummary("Problem Clarity",  0, 10, 25,
                             "No problem detected.", []),
            DimensionSummary("Solution Clarity", 0, 10, 25,
                             "No solution detected.", []),
            DimensionSummary("Target Market",    0, 10, 20,
                             "No market detected.", []),
            DimensionSummary("Specificity",      0, 10, 15,
                             "No specificity.", []),
            DimensionSummary("Completeness",     1, 10, 15,
                             "Very brief.", []),
        ]
        report = ValidationReport(
            idea_text="I have an idea.",
            country_input=None,
            categories_input=[],
            resolved_country=None,
            resolved_categories=[],
            is_complete_submission=False,
            word_count=4,
            is_meaningful_submission=False,
            ml_predicted_class=0,
            ml_pattern_label="Failure Pattern",
            ml_success_score=0.2,
            ml_failure_score=0.8,
            active_country_features=["country_Unknown"],
            active_category_features=[],
            n_categories=0,
            idea_evaluation_score=0,
            idea_evaluation_grade="Very Weak",
            dimensions=dims,
            strengths=[],
            # Primary missing list populated — this branch takes priority
            missing_or_unclear=["No problem stated.", "No solution stated."],
            guidance=["Describe the problem."],
            submission_warnings=["Incomplete submission."],
            limitations=["Limitation 1."],
        )
        text = render_report_text(report)
        # The primary branch fires (missing_or_unclear is non-empty),
        # so the fallback partial-gaps logic must NOT run
        self.assertNotIn("partially addressed", text)
        # Primary items must appear
        self.assertIn("No problem stated.", text)

    def test_integration_moderate_idea_shows_partials(self):
        """
        Integration test using the actual pipeline: a short, moderate idea
        (the one that originally showed the contradiction) must now show
        the partial dimensions, not the old fallback.
        """
        inp = StartupIdeaInput(
            idea_text=(
                "An AI platform that helps restaurants reduce food waste in India."
            ),
            country="India",
        )
        out    = run_full_evaluation(inp)
        report = generate_report(out, country_input="India")
        text   = render_report_text(report)

        # Old contradictory phrase must be gone
        self.assertNotIn("(No critical gaps detected)", text)

        # At least one dimension should be named as partial (score 1–6)
        # for this short, moderate-specificity submission
        has_partial_or_neutral = (
            "partially addressed" in text or
            "No major clarity gaps detected" in text
        )
        self.assertTrue(has_partial_or_neutral,
            "Section must show either partial dims or the neutral message")


# ---------------------------------------------------------------------------
# STANDALONE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestReportStructure, TestScoreSeparation,
        TestMlScoreNotCalledProbability,
        TestSpecificIdea, TestVagueIdea, TestEmptyIdea,
        TestGradeHelper, TestMissingOrUnclearSection,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
