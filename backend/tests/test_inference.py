"""
tests/test_inference.py
========================
Phase 4A — Inference Pipeline Test Suite

Verifies that the inference pipeline produces valid, consistent outputs
for every required test case.

WHAT IS TESTED
--------------
For every test case:
  1. Script does not crash (no exception)
  2. Feature vector has exactly 65 features
  3. Feature order matches feature_names.json
  4. No NaN values in the feature vector
  5. Predicted class is 0 or 1
  6. Probability scores are in [0.0, 1.0]
  7. prob_success + prob_failure ≈ 1.0
  8. All category flags in the result come from APPROVED_CATEGORIES only
  9. All country flags match APPROVED_COUNTRIES or country_other
 10. country_Unknown and cat_Unknown behave correctly per the submission type

TEST CASES
----------
TEST 1: AI platform for restaurants to reduce food waste in India
TEST 2: Online education platform for competitive exam students in India
TEST 3: Healthcare technology for remote patient monitoring
TEST 4: Software platform for small businesses to manage invoices
TEST 5: Deliberately vague — "I have an app idea"

HOW TO RUN
----------
From the project root:
    python -m pytest tests/test_inference.py -v
or:
    python tests/test_inference.py
"""

import json
import os
import sys
import unittest

# Allow running from project root or tests/ directory
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.feature_extractor import (
    APPROVED_CATEGORIES,
    APPROVED_CATEGORY_SET,
    EXPECTED_FEATURE_COUNT,
    FEATURE_ORDER,
    StartupIdeaInput,
    _category_to_col,
    extract_features,
    feature_vector_to_ordered_list,
)
from inference.model_loader import load_model_and_metadata, predict
from inference.vocabulary import APPROVED_COUNTRIES


class TestInferencePipeline(unittest.TestCase):
    """
    Shared setup: loads the model and feature metadata once for all tests.
    """

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests — avoids reloading per test."""
        cls.model, cls.feature_names, cls.prep_meta = load_model_and_metadata()

    # ------------------------------------------------------------------
    # HELPER ASSERTIONS (reused across all test cases)
    # ------------------------------------------------------------------

    def _run_inference(self, input_data: StartupIdeaInput) -> dict:
        """Run full inference and return the result bundle."""
        result         = extract_features(input_data)
        fv             = result.feature_vector
        ordered_list   = feature_vector_to_ordered_list(fv)
        predicted_class, prob_failure, prob_success = predict(self.model, ordered_list, self.feature_names)
        return {
            "result":          result,
            "fv":              fv,
            "ordered_list":    ordered_list,
            "predicted_class": predicted_class,
            "prob_failure":    prob_failure,
            "prob_success":    prob_success,
        }

    def _assert_feature_vector_valid(self, fv: dict, test_name: str):
        """Core structural assertions applied to every test case."""

        # 1. Correct feature count
        self.assertEqual(
            len(fv), EXPECTED_FEATURE_COUNT,
            f"[{test_name}] Feature count {len(fv)} != {EXPECTED_FEATURE_COUNT}"
        )

        # 2. Feature order matches FEATURE_ORDER (and therefore feature_names.json)
        self.assertEqual(
            list(fv.keys()), list(FEATURE_ORDER),
            f"[{test_name}] Feature order does not match FEATURE_ORDER"
        )

        # 3. Feature order matches feature_names.json loaded at setup
        self.assertEqual(
            list(fv.keys()), self.feature_names,
            f"[{test_name}] Feature keys do not match feature_names.json"
        )

        # 4. No NaN values
        for name, val in fv.items():
            self.assertIsNotNone(
                val,
                f"[{test_name}] NaN in feature '{name}'"
            )
            self.assertIsInstance(
                val, int,
                f"[{test_name}] Non-int value in '{name}': {val!r}"
            )

        # 5. No forbidden columns
        forbidden = {
            "status", "target", "funding_total_usd", "funding_rounds",
            "last_funding_at", "first_funding_at", "founding_year",
            "has_founding_date", "is_usa",
        }
        for col in forbidden:
            self.assertNotIn(
                col, fv,
                f"[{test_name}] Forbidden column '{col}' present in feature vector"
            )

        # 6. Target column absent
        self.assertNotIn("target", fv, f"[{test_name}] 'target' in feature vector")

    def _assert_prediction_valid(self, predicted_class: int,
                                  prob_failure: float, prob_success: float,
                                  test_name: str):
        """Assertions on the prediction outputs."""

        # Predicted class is 0 or 1
        self.assertIn(
            predicted_class, [0, 1],
            f"[{test_name}] predicted_class {predicted_class} not in {{0, 1}}"
        )

        # Probabilities are in [0, 1]
        self.assertGreaterEqual(prob_success, 0.0,
            f"[{test_name}] prob_success < 0")
        self.assertLessEqual(prob_success, 1.0,
            f"[{test_name}] prob_success > 1")
        self.assertGreaterEqual(prob_failure, 0.0,
            f"[{test_name}] prob_failure < 0")
        self.assertLessEqual(prob_failure, 1.0,
            f"[{test_name}] prob_failure > 1")

        # Probabilities sum to ~1.0
        total = prob_success + prob_failure
        self.assertAlmostEqual(
            total, 1.0, places=5,
            msg=f"[{test_name}] prob_success + prob_failure = {total}, not ~1.0"
        )

    def _assert_categories_from_vocabulary(self, result, test_name: str):
        """All resolved categories must be in APPROVED_CATEGORIES."""
        for cat in result.resolved_categories:
            self.assertIn(
                cat, APPROVED_CATEGORY_SET,
                f"[{test_name}] Category '{cat}' not in APPROVED_CATEGORIES"
            )

    def _assert_country_flags_consistent(self, fv: dict, test_name: str):
        """
        At most one of {country_X, country_Unknown} can be 1.
        country_other and country_Unknown are mutually exclusive.
        """
        country_flag_cols = [k for k in fv if k.startswith("country_")]
        active = [c for c in country_flag_cols if fv[c] == 1]

        # country_Unknown and a specific country should not both be 1
        specific = [c for c in active if c not in ("country_Unknown", "country_other")]
        if "country_Unknown" in active:
            self.assertEqual(
                len(specific), 0,
                f"[{test_name}] country_Unknown=1 but also specific country flags: {specific}"
            )

        # country_Unknown and country_other should not both be 1
        both_unknown_and_other = (
            "country_Unknown" in active and "country_other" in active
        )
        self.assertFalse(
            both_unknown_and_other,
            f"[{test_name}] Both country_Unknown and country_other are 1"
        )

    # ------------------------------------------------------------------
    # TEST 1
    # AI platform helping small restaurants reduce food waste in India
    # ------------------------------------------------------------------

    def test_01_india_restaurant_ai(self):
        """
        Test 1: AI platform helping small restaurants reduce food waste in India.
        Expected: country=IND, categories include Analytics and/or Clean Technology,
                  is_complete_submission=True
        """
        test_name  = "TEST_1_India_Restaurant_AI"
        input_data = StartupIdeaInput(
            idea_text = (
                "AI platform helping small restaurants reduce food waste in India."
            ),
            country   = "India",
        )

        out = self._run_inference(input_data)
        fv  = out["fv"]

        # Structural checks
        self._assert_feature_vector_valid(fv, test_name)
        self._assert_prediction_valid(
            out["predicted_class"], out["prob_failure"],
            out["prob_success"], test_name
        )
        self._assert_categories_from_vocabulary(out["result"], test_name)
        self._assert_country_flags_consistent(fv, test_name)

        # Specific expectations
        self.assertEqual(fv["country_IND"], 1,
            f"[{test_name}] country_IND should be 1 for India")
        self.assertEqual(fv["country_Unknown"], 0,
            f"[{test_name}] country_Unknown should be 0 (country was provided)")

        # At least one category should have been matched (AI or food waste or similar)
        active_cats = [k for k in fv if k.startswith("cat_") and fv[k] == 1
                       and k != "cat_Unknown"]
        self.assertGreater(
            len(active_cats), 0,
            f"[{test_name}] No category flags set — expected at least one match"
        )
        self.assertEqual(fv["cat_Unknown"], 0,
            f"[{test_name}] cat_Unknown should be 0 (categories were matched)")

        self.assertTrue(
            out["result"].is_complete_submission,
            f"[{test_name}] is_complete_submission should be True"
        )

        print(f"\n  {test_name}")
        print(f"    Country       : {out['result'].resolved_country}")
        print(f"    Categories    : {out['result'].resolved_categories}")
        print(f"    Predicted     : {out['predicted_class']}")
        print(f"    Success score : {out['prob_success']:.4f}")

    # ------------------------------------------------------------------
    # TEST 2
    # Online education platform for competitive exam students in India
    # ------------------------------------------------------------------

    def test_02_india_education(self):
        """
        Test 2: Online education platform for students preparing for
        competitive exams in India.
        Expected: country=IND, Education category matched.
        """
        test_name  = "TEST_2_India_Education"
        input_data = StartupIdeaInput(
            idea_text = (
                "Online education platform for students preparing for "
                "competitive exams in India."
            ),
            country   = "India",
        )

        out = self._run_inference(input_data)
        fv  = out["fv"]

        self._assert_feature_vector_valid(fv, test_name)
        self._assert_prediction_valid(
            out["predicted_class"], out["prob_failure"],
            out["prob_success"], test_name
        )
        self._assert_categories_from_vocabulary(out["result"], test_name)
        self._assert_country_flags_consistent(fv, test_name)

        self.assertEqual(fv["country_IND"], 1,
            f"[{test_name}] country_IND should be 1")
        self.assertEqual(fv["country_Unknown"], 0,
            f"[{test_name}] country_Unknown should be 0")

        # Education should be matched
        self.assertEqual(fv["cat_Education"], 1,
            f"[{test_name}] cat_Education should be 1 for education platform")
        self.assertEqual(fv["cat_Unknown"], 0,
            f"[{test_name}] cat_Unknown should be 0")
        self.assertTrue(out["result"].is_complete_submission,
            f"[{test_name}] should be complete")

        print(f"\n  {test_name}")
        print(f"    Country       : {out['result'].resolved_country}")
        print(f"    Categories    : {out['result'].resolved_categories}")
        print(f"    Predicted     : {out['predicted_class']}")
        print(f"    Success score : {out['prob_success']:.4f}")

    # ------------------------------------------------------------------
    # TEST 3
    # Healthcare technology platform for remote patient monitoring
    # ------------------------------------------------------------------

    def test_03_healthcare_remote_monitoring(self):
        """
        Test 3: Healthcare technology platform for remote patient monitoring.
        No country provided — tests country_Unknown behavior.
        """
        test_name  = "TEST_3_Healthcare_NoCountry"
        input_data = StartupIdeaInput(
            idea_text = (
                "Healthcare technology platform for remote patient monitoring."
            ),
            # No country provided deliberately
        )

        out = self._run_inference(input_data)
        fv  = out["fv"]

        self._assert_feature_vector_valid(fv, test_name)
        self._assert_prediction_valid(
            out["predicted_class"], out["prob_failure"],
            out["prob_success"], test_name
        )
        self._assert_categories_from_vocabulary(out["result"], test_name)
        self._assert_country_flags_consistent(fv, test_name)

        # No country -> country_Unknown = 1, all specific country flags = 0
        self.assertEqual(fv["country_Unknown"], 1,
            f"[{test_name}] country_Unknown should be 1 (no country given)")
        specific_country_cols = [
            k for k in fv
            if k.startswith("country_") and k not in ("country_Unknown", "country_other")
        ]
        for col in specific_country_cols:
            self.assertEqual(fv[col], 0,
                f"[{test_name}] {col} should be 0 when country is unknown")

        # Healthcare keywords should map to Medical or Health Care
        active_cats = [k for k in fv if k.startswith("cat_") and fv[k] == 1
                       and k != "cat_Unknown"]
        self.assertGreater(len(active_cats), 0,
            f"[{test_name}] At least one category should match for healthcare idea")
        self.assertEqual(fv["cat_Unknown"], 0,
            f"[{test_name}] cat_Unknown should be 0 (healthcare keywords matched)")

        # Submission is incomplete because no country
        self.assertFalse(out["result"].is_complete_submission,
            f"[{test_name}] is_complete_submission should be False (no country)")

        print(f"\n  {test_name}")
        print(f"    Country       : {out['result'].resolved_country} (none)")
        print(f"    Categories    : {out['result'].resolved_categories}")
        print(f"    Predicted     : {out['predicted_class']}")
        print(f"    Success score : {out['prob_success']:.4f}")

    # ------------------------------------------------------------------
    # TEST 4
    # Software platform for small businesses to manage invoices
    # ------------------------------------------------------------------

    def test_04_invoice_software_usa(self):
        """
        Test 4: Software platform for small businesses to manage invoices.
        Uses explicit country and categories to verify exact mapping.
        """
        test_name  = "TEST_4_Invoice_Software_USA"
        input_data = StartupIdeaInput(
            idea_text  = "A software platform for small businesses to manage invoices.",
            country    = "USA",
            categories = ["Software", "Finance"],
        )

        out = self._run_inference(input_data)
        fv  = out["fv"]

        self._assert_feature_vector_valid(fv, test_name)
        self._assert_prediction_valid(
            out["predicted_class"], out["prob_failure"],
            out["prob_success"], test_name
        )
        self._assert_categories_from_vocabulary(out["result"], test_name)
        self._assert_country_flags_consistent(fv, test_name)

        # Explicit USA
        self.assertEqual(fv["country_USA"], 1,
            f"[{test_name}] country_USA should be 1")
        self.assertEqual(fv["country_Unknown"], 0,
            f"[{test_name}] country_Unknown should be 0")

        # Explicit categories
        self.assertEqual(fv["cat_Software"], 1,
            f"[{test_name}] cat_Software should be 1 (explicitly provided)")
        self.assertEqual(fv["cat_Finance"], 1,
            f"[{test_name}] cat_Finance should be 1 (explicitly provided)")
        self.assertEqual(fv["cat_Unknown"], 0,
            f"[{test_name}] cat_Unknown should be 0")

        # n_categories >= 2 (at least Software + Finance, possibly more from keywords)
        self.assertGreaterEqual(fv["n_categories"], 2,
            f"[{test_name}] n_categories should be >= 2")

        self.assertTrue(out["result"].is_complete_submission,
            f"[{test_name}] should be complete")

        print(f"\n  {test_name}")
        print(f"    Country       : {out['result'].resolved_country}")
        print(f"    Categories    : {out['result'].resolved_categories}")
        print(f"    n_categories  : {fv['n_categories']}")
        print(f"    Predicted     : {out['predicted_class']}")
        print(f"    Success score : {out['prob_success']:.4f}")

    # ------------------------------------------------------------------
    # TEST 5
    # Deliberately vague idea — "I have an app idea"
    # ------------------------------------------------------------------

    def test_05_vague_idea(self):
        """
        Test 5: Deliberately vague — "I have an app idea."
        The system must NOT invent detailed categories.
        'app' maps to the generic 'Apps' category — that is acceptable.
        No country -> country_Unknown = 1.
        """
        test_name  = "TEST_5_Vague"
        input_data = StartupIdeaInput(
            idea_text = "I have an app idea.",
            # No country, no explicit categories
        )

        out = self._run_inference(input_data)
        fv  = out["fv"]

        self._assert_feature_vector_valid(fv, test_name)
        self._assert_prediction_valid(
            out["predicted_class"], out["prob_failure"],
            out["prob_success"], test_name
        )
        self._assert_categories_from_vocabulary(out["result"], test_name)
        self._assert_country_flags_consistent(fv, test_name)

        # No country
        self.assertEqual(fv["country_Unknown"], 1,
            f"[{test_name}] country_Unknown should be 1 (no country given)")

        # The key requirement: no invented category that isn't in the vocabulary
        for cat in out["result"].resolved_categories:
            self.assertIn(cat, APPROVED_CATEGORY_SET,
                f"[{test_name}] Invented category '{cat}' not in APPROVED_CATEGORIES")

        # Submission is incomplete
        self.assertFalse(out["result"].is_complete_submission,
            f"[{test_name}] should be incomplete (no country, minimal category)")

        # Probabilities must still be valid numbers in range
        self.assertGreaterEqual(out["prob_success"], 0.0)
        self.assertLessEqual(out["prob_success"], 1.0)
        self.assertIn(out["predicted_class"], [0, 1])

        print(f"\n  {test_name}")
        print(f"    Country       : {out['result'].resolved_country} (none)")
        print(f"    Categories    : {out['result'].resolved_categories}")
        print(f"    cat_Unknown   : {fv['cat_Unknown']}")
        print(f"    Predicted     : {out['predicted_class']}")
        print(f"    Success score : {out['prob_success']:.4f}")
        if out["result"].warnings:
            for w in out["result"].warnings:
                print(f"    Warning: {w[:80]}")

    # ------------------------------------------------------------------
    # STRUCTURAL / CONTRACT TESTS
    # ------------------------------------------------------------------

    def test_feature_order_matches_json(self):
        """FEATURE_ORDER in feature_extractor must exactly match feature_names.json."""
        self.assertEqual(
            list(FEATURE_ORDER), self.feature_names,
            "FEATURE_ORDER in feature_extractor.py does not match feature_names.json"
        )

    def test_feature_count_constant(self):
        """EXPECTED_FEATURE_COUNT must equal 65."""
        self.assertEqual(EXPECTED_FEATURE_COUNT, 65)

    def test_model_feature_count_matches(self):
        """Model n_features_in_ must equal EXPECTED_FEATURE_COUNT."""
        self.assertEqual(
            self.model.n_features_in_, EXPECTED_FEATURE_COUNT,
            f"Model expects {self.model.n_features_in_} features; "
            f"expected {EXPECTED_FEATURE_COUNT}"
        )

    def test_model_classes_are_0_and_1(self):
        """Model classes must be [0, 1]."""
        import numpy as np
        self.assertTrue(
            np.array_equal(self.model.classes_, [0, 1]),
            f"Model classes: {self.model.classes_}"
        )

    def test_category_col_mapping(self):
        """
        Verify _category_to_col produces names that all exist in FEATURE_ORDER.
        Specifically tests the non-trivial cases.
        """
        cases = {
            "Hardware + Software": "cat_Hardware_plus_Software",
            "E-Commerce":          "cat_E-Commerce",
            "Health and Wellness": "cat_Health_and_Wellness",
            "SaaS":                "cat_SaaS",
            "FinTech":             "cat_FinTech",
            "Clean Technology":    "cat_Clean_Technology",
        }
        for cat_name, expected_col in cases.items():
            actual_col = _category_to_col(cat_name)
            self.assertEqual(
                actual_col, expected_col,
                f"_category_to_col('{cat_name}') = '{actual_col}' != '{expected_col}'"
            )
            self.assertIn(
                actual_col, FEATURE_ORDER,
                f"Column '{actual_col}' not in FEATURE_ORDER"
            )

    def test_explicit_invalid_category_is_rejected(self):
        """An explicit category not in the approved vocabulary must be ignored."""
        input_data = StartupIdeaInput(
            idea_text  = "A platform for something.",
            country    = "USA",
            categories = ["FoodTech", "AgriTech"],   # not in vocab
        )
        result = extract_features(input_data)
        for cat in result.resolved_categories:
            self.assertIn(
                cat, APPROVED_CATEGORY_SET,
                f"Invalid category '{cat}' slipped through validation"
            )
        self.assertTrue(
            len(result.warnings) > 0,
            "Expected warnings for unrecognised categories"
        )

    def test_zero_features_for_empty_idea(self):
        """
        An empty idea with no country should produce a valid (though
        mostly-zero) feature vector — the pipeline should not crash.
        """
        input_data = StartupIdeaInput(idea_text="", country=None)
        result = extract_features(input_data)
        fv     = result.feature_vector
        self.assertEqual(len(fv), EXPECTED_FEATURE_COUNT)
        for val in fv.values():
            self.assertIsNotNone(val)

    def test_country_other_for_unknown_country(self):
        """A country not in the top-10 list sets country_other=1."""
        input_data = StartupIdeaInput(
            idea_text = "A startup in Brazil.",
            country   = "Brazil",
        )
        result = extract_features(input_data)
        fv     = result.feature_vector
        # Brazil is not in top-10 -> country_other=1, country_Unknown=0
        self.assertEqual(fv["country_other"], 1,
            "Brazil (not in top-10) should set country_other=1")
        self.assertEqual(fv["country_Unknown"], 0,
            "country_Unknown should be 0 when country was provided (even if not top-10)")

    def test_mutual_exclusion_country_flags(self):
        """For any input, no more than one specific country flag can be 1."""
        input_data = StartupIdeaInput(
            idea_text = "A software platform in Germany.",
            country   = "Germany",
        )
        result = extract_features(input_data)
        fv     = result.feature_vector
        specific = [
            k for k in fv
            if k.startswith("country_")
            and k not in ("country_Unknown", "country_other")
            and fv[k] == 1
        ]
        self.assertLessEqual(
            len(specific), 1,
            f"Multiple specific country flags active: {specific}"
        )
        self.assertEqual(fv["country_DEU"], 1,
            "DEU should be 1 for Germany")

    # ------------------------------------------------------------------
    # REGRESSION TESTS FOR D1 AND D2 FIXES (Phase 4A audit fixes)
    # ------------------------------------------------------------------

    def test_d1_invalid_column_raises_value_error(self):
        """
        D1 fix: if _category_to_col() produces a column not in FEATURE_ORDER,
        extract_features() must raise ValueError, not silently ignore it.

        We simulate the failure by monkey-patching _resolve_categories to
        inject a fake category name that cannot exist in FEATURE_ORDER.
        """
        import inference.feature_extractor as fe

        original_resolve = fe._resolve_categories

        def _inject_bad_category(explicit_categories, idea_text, log,
                                  warnings, unresolved):
            result = original_resolve(
                explicit_categories, idea_text, log, warnings, unresolved
            )
            # Inject a category whose column name will not be in FEATURE_ORDER
            result.append("__NonExistentCategory__")
            return result

        fe._resolve_categories = _inject_bad_category
        try:
            inp = StartupIdeaInput(idea_text="test idea", country="USA")
            with self.assertRaises(ValueError) as ctx:
                extract_features(inp)
            self.assertIn(
                "__NonExistentCategory__", str(ctx.exception),
                "ValueError message should name the missing category"
            )
        finally:
            # Always restore the original function
            fe._resolve_categories = original_resolve

    def test_d1_all_approved_categories_produce_valid_columns(self):
        """
        D1 fix integrity check: every category in APPROVED_CATEGORIES must
        map to a column that exists in FEATURE_ORDER.
        If this fails, D1's ValueError would fire on every real submission.
        """
        for cat in APPROVED_CATEGORIES:
            col = _category_to_col(cat)
            self.assertIn(
                col, FEATURE_ORDER,
                f"APPROVED_CATEGORIES member '{cat}' maps to '{col}' "
                f"which is NOT in FEATURE_ORDER — D1 fix would block all "
                f"submissions using this category"
            )

    def test_d2_cat_technology_keyword_activates(self):
        """
        D2 fix: 'tech company', 'tech startup', 'tech firm', 'tech solutions',
        'tech services', 'tech platform' must activate cat_Technology.
        cat_Technology is a real trained feature (position 28 / 65).
        """
        tech_phrases = [
            "we are a tech company",
            "a tech startup building products",
            "a tech firm in San Francisco",
            "tech solutions for enterprise clients",
            "tech services platform",
            "a tech platform for developers",
        ]
        for phrase in tech_phrases:
            inp = StartupIdeaInput(idea_text=phrase, country="USA")
            res = extract_features(inp)
            self.assertEqual(
                res.feature_vector["cat_Technology"], 1,
                f"cat_Technology should be 1 for: '{phrase}'"
            )
            self.assertIn(
                "Technology", res.resolved_categories,
                f"'Technology' should be in resolved_categories for: '{phrase}'"
            )

    def test_d2_technology_keywords_do_not_shadow_clean_technology(self):
        """
        D2 fix safety: 'clean technology company/startup' must NOT activate
        cat_Technology. The new keywords use 'tech *' prefix only, which
        cannot appear as a substring of 'clean technology X'.
        """
        non_shadow_phrases = [
            "a clean technology company",
            "a clean technology startup",
            "clean technology solutions provider",
        ]
        for phrase in non_shadow_phrases:
            inp = StartupIdeaInput(idea_text=phrase, country="USA")
            res = extract_features(inp)
            self.assertEqual(
                res.feature_vector["cat_Technology"], 0,
                f"cat_Technology should be 0 for: '{phrase}' "
                f"(must not shadow Clean Technology)"
            )
            self.assertEqual(
                res.feature_vector["cat_Clean_Technology"], 1,
                f"cat_Clean_Technology should still be 1 for: '{phrase}'"
            )

    def test_d2_technology_keywords_do_not_shadow_information_technology(self):
        """
        D2 fix safety: 'information technology services/solutions' must NOT
        activate cat_Technology.
        """
        non_shadow_phrases = [
            "an information technology services company",
            "information technology solutions provider",
        ]
        for phrase in non_shadow_phrases:
            inp = StartupIdeaInput(idea_text=phrase, country="USA")
            res = extract_features(inp)
            self.assertEqual(
                res.feature_vector["cat_Technology"], 0,
                f"cat_Technology should be 0 for: '{phrase}'"
            )

    def test_d2_explicit_technology_category_still_works(self):
        """
        D2 fix: explicit categories=['Technology'] must still activate
        cat_Technology, independent of keyword matching.
        """
        inp = StartupIdeaInput(
            idea_text  = "Our startup is difficult to categorise.",
            country    = "USA",
            categories = ["Technology"],
        )
        res = extract_features(inp)
        self.assertEqual(
            res.feature_vector["cat_Technology"], 1,
            "Explicit categories=['Technology'] must activate cat_Technology"
        )

    def test_five_original_test_cases_still_pass(self):
        """
        Regression: the five original Phase 4A test-case inputs must still
        produce the same structural properties after the D1 and D2 fixes.
        """
        cases = [
            # (idea, country, cats, expected_country_flag,
            #  expected_cat_flags_nonzero, complete)
            (
                "AI platform helping small restaurants reduce food waste in India.",
                "India", [],
                "country_IND",
                ["cat_Analytics", "cat_Clean_Technology", "cat_Services"],
                True,
            ),
            (
                "Online education platform for students preparing for "
                "competitive exams in India.",
                "India", [],
                "country_IND",
                ["cat_Education"],
                True,
            ),
            (
                "Healthcare technology platform for remote patient monitoring.",
                None, [],
                "country_Unknown",
                ["cat_Health_Care", "cat_Medical"],
                False,
            ),
            (
                "A software platform for small businesses to manage invoices.",
                "USA", ["Software", "Finance"],
                "country_USA",
                ["cat_Software", "cat_Finance"],
                True,
            ),
            (
                "I have an app idea.",
                None, [],
                "country_Unknown",
                ["cat_Apps"],
                False,
            ),
        ]

        for idea, country, cats, expected_country, expected_cats, complete in cases:
            inp = StartupIdeaInput(idea_text=idea, country=country, categories=cats)
            res = extract_features(inp)
            fv  = res.feature_vector

            with self.subTest(idea=idea[:50]):
                # Feature count always 65
                self.assertEqual(len(fv), EXPECTED_FEATURE_COUNT)
                # Expected country flag is 1
                self.assertEqual(
                    fv[expected_country], 1,
                    f"Expected {expected_country}=1 for: '{idea[:50]}'"
                )
                # All expected category flags are 1
                for cat_col in expected_cats:
                    self.assertEqual(
                        fv[cat_col], 1,
                        f"Expected {cat_col}=1 for: '{idea[:50]}'"
                    )
                # cat_Unknown is 0 when real categories matched
                if expected_cats:
                    self.assertEqual(
                        fv["cat_Unknown"], 0,
                        f"cat_Unknown should be 0 when categories were matched"
                    )
                # Completeness
                self.assertEqual(
                    res.is_complete_submission, complete,
                    f"is_complete_submission should be {complete} for: '{idea[:50]}'"
                )


# ---------------------------------------------------------------------------
# STANDALONE EXECUTION (without pytest)
# ---------------------------------------------------------------------------

def run_all_tests_standalone():
    """Run tests and print results without pytest."""
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromTestCase(TestInferencePipeline)
    runner  = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result  = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests_standalone()
    sys.exit(0 if success else 1)
