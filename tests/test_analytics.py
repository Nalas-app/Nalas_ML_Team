"""
test_analytics.py
=================
Unit tests for src/analytics.py

Covers:
  - _numeric_stats (via analyse_dataset): mean, median, std, min, max, empty list
  - analyse_dataset : total count, duplicates, missing rates, top ingredients,
                      tag distribution, avg ingredients / steps
  - export_pipeline_summary : file created, correct keys, elapsed time plausible
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json, tempfile, unittest
from datetime import datetime
from src.analytics import analyse_dataset, export_pipeline_summary


def _recipe(rid, name, items, tags, time, servings,
            imputed_time=False, imputed_svc=False, is_dup=False):
    imp_fields = []
    orig_vals  = {}
    if imputed_time:
        imp_fields.append("prep_time_minutes")
        orig_vals["prep_time_minutes"] = None
    if imputed_svc:
        imp_fields.append("servings")
        orig_vals["servings"] = None
    return {
        "recipe_id":         rid,
        "name":              name,
        "ingredients":       [{"item": it, "original_text": it, "normalized_item": None} for it in items],
        "instructions":      ["Step one.", "Step two.", "Step three."],
        "prep_time_minutes": time,
        "servings":          servings,
        "tags":              tags,
        "_meta": {
            "imputed_fields":  imp_fields,
            "original_values": orig_vals,
            "parse_warnings":  [],
            "is_duplicate":    is_dup,
            "duplicate_of":    "r01" if is_dup else None,
        },
    }


SAMPLE_RECIPES = [
    _recipe("r01", "Tacos",    ["chicken","salsa","tortilla"],  ["quick","protein-rich"], time=15,  servings=4),
    _recipe("r02", "Pancakes", ["flour","milk","egg","sugar"],   ["quick","baking"],        time=20,  servings=2),
    _recipe("r03", "Stew",     ["beef","potato","water"],        ["slow-cook","protein-rich"],time=120, servings=6),
    _recipe("r04", "Salad",    ["lettuce","tomato","dressing"],  ["quick","vegetarian","vegan-friendly"],
            time=22, servings=4, imputed_time=True),
    _recipe("r05", "Cookies",  ["flour","butter","sugar","eggs"],["baking"],               time=35,  servings=48),
    _recipe("r06", "Tacos Dup",["chicken","salsa","tortilla"],   ["quick","protein-rich"],  time=15,  servings=4,
            imputed_svc=True, is_dup=True),
]


# ---------------------------------------------------------------------------
class TestAnalyseDataset(unittest.TestCase):

    def setUp(self):
        self.stats = analyse_dataset(SAMPLE_RECIPES)

    def test_total_recipes(self):
        self.assertEqual(self.stats["total_recipes"], 6)

    def test_duplicate_count(self):
        self.assertEqual(self.stats["duplicate_count"], 1)

    def test_imputed_prep_time_count(self):
        self.assertEqual(self.stats["imputed_prep_time_count"], 1)

    def test_imputed_servings_count(self):
        self.assertEqual(self.stats["imputed_servings_count"], 1)

    def test_missing_rate_prep_time(self):
        # 1 out of 6 → 16.7%
        self.assertAlmostEqual(self.stats["missing_rate_prep_time_pct"], 16.7, places=1)

    def test_missing_rate_servings(self):
        self.assertAlmostEqual(self.stats["missing_rate_servings_pct"], 16.7, places=1)

    def test_prep_time_mean(self):
        times = [15, 20, 120, 22, 35, 15]
        import numpy as np
        expected = round(float(np.mean(times)), 2)
        self.assertAlmostEqual(self.stats["prep_time_minutes_stats"]["mean"], expected, places=1)

    def test_prep_time_median(self):
        import numpy as np
        times = [15, 20, 120, 22, 35, 15]
        expected = round(float(np.median(times)), 2)
        self.assertAlmostEqual(self.stats["prep_time_minutes_stats"]["median"], expected, places=1)

    def test_prep_time_min_max(self):
        self.assertEqual(self.stats["prep_time_minutes_stats"]["min"], 15)
        self.assertEqual(self.stats["prep_time_minutes_stats"]["max"], 120)

    def test_servings_stats_present(self):
        s = self.stats["servings_stats"]
        for key in ("count", "mean", "median", "std", "min", "max"):
            self.assertIn(key, s)

    def test_avg_ingredients_per_recipe(self):
        # r01:3, r02:4, r03:3, r04:3, r05:4, r06:3 → avg 3.333
        self.assertAlmostEqual(self.stats["avg_ingredients_per_recipe"], 3.33, places=1)

    def test_avg_steps_per_recipe(self):
        # all have 3 steps
        self.assertAlmostEqual(self.stats["avg_steps_per_recipe"], 3.0, places=1)

    def test_top_ingredients_is_list(self):
        self.assertIsInstance(self.stats["top_15_ingredients"], list)

    def test_top_ingredients_have_item_and_count(self):
        for entry in self.stats["top_15_ingredients"]:
            self.assertIn("item",  entry)
            self.assertIn("count", entry)

    def test_top_ingredients_sorted_by_count_desc(self):
        counts = [e["count"] for e in self.stats["top_15_ingredients"]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_flour_in_top_ingredients(self):
        # flour appears in r02 and r05
        items = [e["item"] for e in self.stats["top_15_ingredients"]]
        self.assertIn("flour", items)

    def test_chicken_frequency(self):
        items_map = {e["item"]: e["count"] for e in self.stats["top_15_ingredients"]}
        self.assertEqual(items_map.get("chicken", 0), 2)

    def test_tag_distribution_is_dict(self):
        self.assertIsInstance(self.stats["tag_distribution"], dict)

    def test_tag_distribution_contains_quick(self):
        self.assertIn("quick", self.stats["tag_distribution"])

    def test_quick_count(self):
        # quick in r01,r02,r04,r06 → 4
        self.assertEqual(self.stats["tag_distribution"]["quick"], 4)

    def test_empty_dataset(self):
        stats = analyse_dataset([])
        self.assertEqual(stats["total_recipes"], 0)
        self.assertEqual(stats["top_15_ingredients"], [])
        self.assertEqual(stats["tag_distribution"], {})


# ---------------------------------------------------------------------------
class TestExportPipelineSummary(unittest.TestCase):

    def test_file_created(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        stats = analyse_dataset(SAMPLE_RECIPES)
        export_pipeline_summary(SAMPLE_RECIPES, [], stats, datetime.now(), path=path)
        self.assertTrue(os.path.exists(path))
        os.unlink(path)

    def test_file_is_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        stats = analyse_dataset(SAMPLE_RECIPES)
        export_pipeline_summary(SAMPLE_RECIPES, [], stats, datetime.now(), path=path)
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        os.unlink(path)

    def test_summary_has_required_keys(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        stats = analyse_dataset(SAMPLE_RECIPES)
        export_pipeline_summary(SAMPLE_RECIPES, [], stats, datetime.now(), path=path)
        with open(path) as f:
            data = json.load(f)
        for key in ("run_timestamp", "elapsed_seconds", "total_recipes",
                    "validation_issues", "dataset_stats"):
            self.assertIn(key, data)
        os.unlink(path)

    def test_total_recipes_correct(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        stats = analyse_dataset(SAMPLE_RECIPES)
        export_pipeline_summary(SAMPLE_RECIPES, [], stats, datetime.now(), path=path)
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["total_recipes"], len(SAMPLE_RECIPES))
        os.unlink(path)

    def test_elapsed_seconds_positive(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        stats = analyse_dataset(SAMPLE_RECIPES)
        export_pipeline_summary(SAMPLE_RECIPES, [], stats, datetime.now(), path=path)
        with open(path) as f:
            data = json.load(f)
        self.assertGreaterEqual(data["elapsed_seconds"], 0)
        os.unlink(path)

    def test_validation_issues_counts(self):
        report = [
            {"severity": "error"},
            {"severity": "error"},
            {"severity": "warning"},
            {"severity": "info"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        stats = analyse_dataset(SAMPLE_RECIPES)
        export_pipeline_summary(SAMPLE_RECIPES, report, stats, datetime.now(), path=path)
        with open(path) as f:
            data = json.load(f)
        vi = data["validation_issues"]
        self.assertEqual(vi["errors"],   2)
        self.assertEqual(vi["warnings"], 1)
        self.assertEqual(vi["info"],     1)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
