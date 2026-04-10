"""
test_imputation.py
==================
Unit tests for src/imputation.py

Covers:
  - impute_missing_values : all present, missing time, missing servings,
                             both missing, multiple recipes, median calculation
  - build_imputation_log  : only imputed recipes logged, correct fields
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from src.imputation import impute_missing_values, build_imputation_log
from src.config import DEFAULT_SERVINGS


def _recipe(rid, time=None, servings=None):
    return {
        "recipe_id":         rid,
        "name":              f"Recipe {rid}",
        "ingredients":       [{"item": "flour", "original_text": "flour"}],
        "instructions":      ["Do something."],
        "prep_time_minutes": time,
        "servings":          servings,
        "tags":              [],
        "_meta": {
            "imputed_fields":  [],
            "original_values": {},
            "parse_warnings":  [],
            "is_duplicate":    False,
            "duplicate_of":    None,
        },
    }


# ---------------------------------------------------------------------------
class TestImputeMissingValues(unittest.TestCase):

    def test_no_missing_values_unchanged(self):
        recipes = [_recipe("r01", time=20, servings=4)]
        result  = impute_missing_values(recipes)
        r = result[0]
        self.assertEqual(r["prep_time_minutes"], 20)
        self.assertEqual(r["servings"], 4)
        self.assertEqual(r["_meta"]["imputed_fields"], [])

    def test_missing_servings_gets_default(self):
        recipes = [_recipe("r01", time=20, servings=None)]
        result  = impute_missing_values(recipes)
        self.assertEqual(result[0]["servings"], DEFAULT_SERVINGS)

    def test_missing_servings_logged_in_meta(self):
        recipes = [_recipe("r01", time=20, servings=None)]
        result  = impute_missing_values(recipes)
        self.assertIn("servings", result[0]["_meta"]["imputed_fields"])

    def test_original_servings_recorded_as_none(self):
        recipes = [_recipe("r01", time=20, servings=None)]
        result  = impute_missing_values(recipes)
        self.assertIsNone(result[0]["_meta"]["original_values"]["servings"])

    def test_missing_time_gets_median(self):
        # 3 known: 10, 20, 30 → median 20
        recipes = [
            _recipe("r01", time=10, servings=4),
            _recipe("r02", time=20, servings=4),
            _recipe("r03", time=30, servings=4),
            _recipe("r04", time=None, servings=4),   # missing
        ]
        result = impute_missing_values(recipes)
        self.assertEqual(result[3]["prep_time_minutes"], 20)

    def test_missing_time_logged_in_meta(self):
        recipes = [_recipe("r01", time=20, servings=4),
                   _recipe("r02", time=None, servings=4)]
        result  = impute_missing_values(recipes)
        self.assertIn("prep_time_minutes", result[1]["_meta"]["imputed_fields"])

    def test_original_time_recorded_as_none(self):
        recipes = [_recipe("r01", time=20, servings=4),
                   _recipe("r02", time=None, servings=4)]
        result  = impute_missing_values(recipes)
        self.assertIsNone(result[1]["_meta"]["original_values"]["prep_time_minutes"])

    def test_both_missing_both_imputed(self):
        recipes = [_recipe("r01", time=20, servings=4),
                   _recipe("r02", time=None, servings=None)]
        result  = impute_missing_values(recipes)
        r = result[1]
        self.assertIsNotNone(r["prep_time_minutes"])
        self.assertEqual(r["servings"], DEFAULT_SERVINGS)
        self.assertIn("prep_time_minutes", r["_meta"]["imputed_fields"])
        self.assertIn("servings",          r["_meta"]["imputed_fields"])

    def test_median_robust_to_outlier(self):
        # Outlier of 600 should not inflate median
        recipes = [
            _recipe("r01", time=10, servings=4),
            _recipe("r02", time=15, servings=4),
            _recipe("r03", time=600, servings=4),   # outlier
            _recipe("r04", time=None, servings=4),  # missing
        ]
        result = impute_missing_values(recipes)
        imputed_time = result[3]["prep_time_minutes"]
        # Median of [10, 15, 600] = 15; should not be 208 (mean)
        self.assertLessEqual(imputed_time, 20)

    def test_all_missing_time_falls_back_to_30(self):
        # If ALL times are missing, median can't be computed
        # Function should fallback to default (30)
        recipes = [_recipe("r01", time=None, servings=4)]
        result  = impute_missing_values(recipes)
        # Should not crash, and should get a numeric value
        self.assertIsNotNone(result[0]["prep_time_minutes"])
        self.assertIsInstance(result[0]["prep_time_minutes"], (int, float))

    def test_known_values_not_overwritten(self):
        recipes = [_recipe("r01", time=45, servings=6)]
        result  = impute_missing_values(recipes)
        self.assertEqual(result[0]["prep_time_minutes"], 45)
        self.assertEqual(result[0]["servings"],          6)

    def test_imputation_is_integer(self):
        recipes = [_recipe("r01", time=10, servings=4),
                   _recipe("r02", time=21, servings=4),  # median = 15.5 → 16
                   _recipe("r03", time=None, servings=4)]
        result  = impute_missing_values(recipes)
        self.assertIsInstance(result[2]["prep_time_minutes"], int)


# ---------------------------------------------------------------------------
class TestBuildImputationLog(unittest.TestCase):

    def test_empty_when_nothing_imputed(self):
        recipes = [_recipe("r01", time=20, servings=4)]
        impute_missing_values(recipes)
        log = build_imputation_log(recipes)
        self.assertEqual(log, [])

    def test_entry_created_for_imputed_recipe(self):
        recipes = [_recipe("r01", time=20, servings=4),
                   _recipe("r02", time=None, servings=None)]
        impute_missing_values(recipes)
        log = build_imputation_log(recipes)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["recipe_id"], "r02")

    def test_log_entry_has_required_keys(self):
        recipes = [_recipe("r01", time=20, servings=4),
                   _recipe("r02", time=None, servings=None)]
        impute_missing_values(recipes)
        log = build_imputation_log(recipes)
        for key in ("recipe_id", "recipe_name", "imputed_fields", "original_values", "final_values"):
            self.assertIn(key, log[0])

    def test_final_values_not_none(self):
        recipes = [_recipe("r01", time=20, servings=4),
                   _recipe("r02", time=None, servings=None)]
        impute_missing_values(recipes)
        log = build_imputation_log(recipes)
        for field, val in log[0]["final_values"].items():
            self.assertIsNotNone(val, f"final_values[{field}] should not be None")

    def test_original_values_all_none(self):
        recipes = [_recipe("r01", time=20, servings=4),
                   _recipe("r02", time=None, servings=None)]
        impute_missing_values(recipes)
        log = build_imputation_log(recipes)
        for field, val in log[0]["original_values"].items():
            self.assertIsNone(val, f"original_values[{field}] should be None")


if __name__ == "__main__":
    unittest.main()
