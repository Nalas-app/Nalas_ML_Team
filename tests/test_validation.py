"""
test_validation.py
==================
Unit tests for src/validation.py

Covers:
  - check_outliers : normal values, time > threshold, servings out of range,
                     high ingredient count
  - validate_all   : clean recipe passes, missing required fields,
                     missing optional fields, duplicate flagged, outlier flagged,
                     parse warnings passed through, schema type mismatch
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json, unittest
from src.validation import check_outliers, validate_all


# ---------------------------------------------------------------------------
# Minimal valid schema for testing
# ---------------------------------------------------------------------------
SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "recipe_id":         {"type": "string"},
        "name":              {"type": "string", "minLength": 1},
        "ingredients":       {"type": "array",  "minItems": 1,
                              "items": {"type": "object",
                                        "properties": {"item": {"type": "string"},
                                                       "original_text": {"type": "string"}},
                                        "required": ["item", "original_text"]}},
        "instructions":      {"type": "array",  "minItems": 1,
                              "items": {"type": "string"}},
        "prep_time_minutes": {"type": ["number", "null"]},
        "servings":          {"type": ["number", "null"]},
    },
    "required": ["recipe_id", "name", "ingredients", "instructions"],
}


def _clean_recipe(**overrides):
    base = {
        "recipe_id":         "r01",
        "name":              "Test Recipe",
        "ingredients":       [{"item": "flour", "quantity": 2.0, "unit": "cup",
                               "original_text": "2 cups flour"}],
        "instructions":      ["Mix and bake."],
        "prep_time_minutes": 30,
        "servings":          4,
        "tags":              [],
        "_meta": {
            "imputed_fields":  [],
            "original_values": {},
            "parse_warnings":  [],
            "is_duplicate":    False,
            "duplicate_of":    None,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
class TestCheckOutliers(unittest.TestCase):

    def test_normal_recipe_no_outliers(self):
        r = _clean_recipe(prep_time_minutes=30, servings=4)
        self.assertEqual(check_outliers(r), [])

    def test_prep_time_over_threshold(self):
        r = _clean_recipe(prep_time_minutes=700)
        issues = check_outliers(r)
        self.assertTrue(any("prep_time_minutes" in i for i in issues))

    def test_prep_time_exactly_at_threshold_ok(self):
        r = _clean_recipe(prep_time_minutes=600)
        issues = check_outliers(r)
        self.assertFalse(any("prep_time_minutes" in i for i in issues))

    def test_servings_above_max(self):
        r = _clean_recipe(servings=150)
        issues = check_outliers(r)
        self.assertTrue(any("servings" in i for i in issues))

    def test_servings_below_min(self):
        r = _clean_recipe(servings=0)
        issues = check_outliers(r)
        self.assertTrue(any("servings" in i for i in issues))

    def test_servings_exactly_at_min_ok(self):
        r = _clean_recipe(servings=1)
        issues = check_outliers(r)
        self.assertFalse(any("below" in i for i in issues))

    def test_high_ingredient_count(self):
        r = _clean_recipe()
        r["ingredients"] = [{"item": f"ing{i}", "original_text": f"ing{i}"} for i in range(35)]
        issues = check_outliers(r)
        self.assertTrue(any("ingredient count" in i for i in issues))

    def test_none_time_ignored(self):
        r = _clean_recipe(prep_time_minutes=None)
        issues = check_outliers(r)
        self.assertFalse(any("prep_time_minutes" in i for i in issues))

    def test_none_servings_ignored(self):
        r = _clean_recipe(servings=None)
        issues = check_outliers(r)
        self.assertFalse(any("servings" in i for i in issues))


# ---------------------------------------------------------------------------
class TestValidateAll(unittest.TestCase):

    def _get_rows(self, recipes, check_type=None, severity=None):
        rows = validate_all(recipes, SCHEMA)
        if check_type:
            rows = [r for r in rows if r["check_type"] == check_type]
        if severity:
            rows = [r for r in rows if r["severity"] == severity]
        return rows

    def test_clean_recipe_no_errors(self):
        rows = validate_all([_clean_recipe()], SCHEMA)
        errors = [r for r in rows if r["severity"] == "error"]
        self.assertEqual(errors, [])

    def test_missing_name_flagged(self):
        r = _clean_recipe()
        r["name"] = ""
        rows = self._get_rows([r], check_type="missing_required")
        fields = [row["field"] for row in rows]
        self.assertIn("name", fields)

    def test_missing_ingredients_flagged(self):
        r = _clean_recipe()
        r["ingredients"] = []
        rows = self._get_rows([r], check_type="missing_required")
        fields = [row["field"] for row in rows]
        self.assertIn("ingredients", fields)

    def test_missing_instructions_flagged(self):
        r = _clean_recipe()
        r["instructions"] = []
        rows = self._get_rows([r], check_type="missing_required")
        fields = [row["field"] for row in rows]
        self.assertIn("instructions", fields)

    def test_missing_prep_time_is_warning_not_error(self):
        r = _clean_recipe(prep_time_minutes=None)
        rows = self._get_rows([r], check_type="missing_optional")
        self.assertTrue(len(rows) > 0)
        self.assertTrue(all(row["severity"] == "warning" for row in rows))

    def test_missing_servings_is_warning(self):
        r = _clean_recipe(servings=None)
        rows = self._get_rows([r], check_type="missing_optional")
        fields = [row["field"] for row in rows]
        self.assertIn("servings", fields)

    def test_duplicate_flagged(self):
        r = _clean_recipe()
        r["_meta"]["is_duplicate"] = True
        r["_meta"]["duplicate_of"] = "r00"
        rows = self._get_rows([r], check_type="duplicate")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["severity"], "warning")

    def test_outlier_flagged(self):
        r = _clean_recipe(prep_time_minutes=800)
        rows = self._get_rows([r], check_type="outlier")
        self.assertTrue(len(rows) > 0)

    def test_parse_warnings_included(self):
        r = _clean_recipe()
        r["_meta"]["parse_warnings"] = ["Prep time not found in text."]
        rows = self._get_rows([r], check_type="parse_warning")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["severity"], "info")

    def test_report_row_has_required_keys(self):
        r = _clean_recipe(prep_time_minutes=None)
        rows = validate_all([r], SCHEMA)
        for row in rows:
            for key in ("recipe_id", "recipe_name", "check_type", "severity", "field", "description"):
                self.assertIn(key, row)

    def test_multiple_recipes_all_validated(self):
        r1 = _clean_recipe(recipe_id="r01")
        r2 = _clean_recipe(recipe_id="r02")
        r2["name"] = ""
        rows = validate_all([r1, r2], SCHEMA)
        ids_with_errors = {r["recipe_id"] for r in rows if r["severity"] == "error"}
        self.assertIn("r02", ids_with_errors)
        self.assertNotIn("r01", ids_with_errors)

    def test_schema_type_mismatch_detected(self):
        r = _clean_recipe()
        r["servings"] = "four"   # should be number or null
        rows = self._get_rows([r], check_type="schema")
        self.assertTrue(len(rows) > 0)


if __name__ == "__main__":
    unittest.main()
