"""
test_enrichment.py
==================
Unit tests for src/enrichment.py

Covers:
  - convert_ingredient_to_si : known units, unknown, no quantity
  - convert_recipe_to_si     : full recipe pass
  - generate_tags            : quick, slow-cook, baking, vegan, protein, blended
  - content_hash             : deterministic, collision-resistance
  - detect_duplicates        : true duplicate, no duplicates, hash-based (ignores text diffs)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from src.enrichment import (
    convert_ingredient_to_si,
    convert_recipe_to_si,
    generate_tags,
    content_hash,
    detect_duplicates,
)


def _make_recipe(name="Test", ingredients=None, instructions=None,
                 prep_time=30, servings=4, recipe_id="r01"):
    return {
        "recipe_id":         recipe_id,
        "name":              name,
        "ingredients":       ingredients or [],
        "instructions":      instructions or ["Do something."],
        "prep_time_minutes": prep_time,
        "servings":          servings,
        "tags":              [],
        "_meta": {"imputed_fields": [], "original_values": {},
                  "parse_warnings": [], "is_duplicate": False, "duplicate_of": None},
    }


# ---------------------------------------------------------------------------
class TestConvertIngredientToSI(unittest.TestCase):

    def test_cup_to_ml(self):
        ing = {"quantity": 1.0, "unit": "cup", "item": "flour", "original_text": "1 cup flour"}
        result = convert_ingredient_to_si(ing)
        self.assertAlmostEqual(result["si_quantity"], 236.588)
        self.assertEqual(result["si_unit"], "ml")

    def test_tbsp_to_ml(self):
        ing = {"quantity": 2.0, "unit": "tbsp", "item": "sugar", "original_text": "2 tbsp sugar"}
        result = convert_ingredient_to_si(ing)
        self.assertAlmostEqual(result["si_quantity"], 29.574)
        self.assertEqual(result["si_unit"], "ml")

    def test_tsp_to_ml(self):
        ing = {"quantity": 1.0, "unit": "tsp", "item": "salt", "original_text": "1 tsp salt"}
        result = convert_ingredient_to_si(ing)
        self.assertAlmostEqual(result["si_quantity"], 4.929)

    def test_oz_to_g(self):
        ing = {"quantity": 4.0, "unit": "oz", "item": "cheese", "original_text": "4 oz cheese"}
        result = convert_ingredient_to_si(ing)
        self.assertAlmostEqual(result["si_quantity"], 113.4, places=0)
        self.assertEqual(result["si_unit"], "g")

    def test_lb_to_g(self):
        ing = {"quantity": 1.0, "unit": "lb", "item": "beef", "original_text": "1 lb beef"}
        result = convert_ingredient_to_si(ing)
        self.assertAlmostEqual(result["si_quantity"], 453.592)

    def test_kg_to_g(self):
        ing = {"quantity": 1.0, "unit": "kg", "item": "potatoes", "original_text": "1 kg potatoes"}
        result = convert_ingredient_to_si(ing)
        self.assertAlmostEqual(result["si_quantity"], 1000.0)

    def test_g_passthrough(self):
        ing = {"quantity": 500.0, "unit": "g", "item": "pasta", "original_text": "500 g pasta"}
        result = convert_ingredient_to_si(ing)
        self.assertAlmostEqual(result["si_quantity"], 500.0)
        self.assertEqual(result["si_unit"], "g")

    def test_ml_passthrough(self):
        ing = {"quantity": 200.0, "unit": "ml", "item": "cream", "original_text": "200 ml cream"}
        result = convert_ingredient_to_si(ing)
        self.assertAlmostEqual(result["si_quantity"], 200.0)

    def test_unknown_unit_no_si_added(self):
        ing = {"quantity": 1.0, "unit": "handful", "item": "nuts", "original_text": "1 handful nuts"}
        result = convert_ingredient_to_si(ing)
        self.assertNotIn("si_quantity", result)
        self.assertNotIn("si_unit", result)

    def test_none_quantity_no_si_added(self):
        ing = {"quantity": None, "unit": "cup", "item": "water", "original_text": "water"}
        result = convert_ingredient_to_si(ing)
        self.assertNotIn("si_quantity", result)

    def test_original_values_not_mutated(self):
        ing = {"quantity": 2.0, "unit": "cup", "item": "flour", "original_text": "2 cups flour"}
        original_qty = ing["quantity"]
        result = convert_ingredient_to_si(ing)
        # Original dict should be unchanged
        self.assertEqual(ing["quantity"], original_qty)
        # Result is a new dict
        self.assertIsNot(result, ing)


# ---------------------------------------------------------------------------
class TestConvertRecipeToSI(unittest.TestCase):

    def test_all_ingredients_processed(self):
        recipe = _make_recipe(ingredients=[
            {"quantity": 2.0, "unit": "cup", "item": "flour", "original_text": "2 cups flour"},
            {"quantity": 1.0, "unit": "lb",  "item": "beef",  "original_text": "1 lb beef"},
            {"quantity": None,"unit": None,   "item": "water", "original_text": "water"},
        ])
        result = convert_recipe_to_si(recipe)
        ings = result["ingredients"]
        self.assertIn("si_quantity", ings[0])
        self.assertIn("si_quantity", ings[1])
        self.assertNotIn("si_quantity", ings[2])  # no qty → no SI


# ---------------------------------------------------------------------------
class TestGenerateTags(unittest.TestCase):

    def _recipe_with(self, items, instructions="", time=None):
        return {
            "name": "Test",
            "ingredients": [{"item": i, "original_text": i} for i in items],
            "instructions": [instructions] if instructions else [],
            "prep_time_minutes": time,
        }

    def test_quick_tag(self):
        r = self._recipe_with(["lettuce"], time=20)
        self.assertIn("quick", generate_tags(r))

    def test_no_quick_tag_over_30(self):
        r = self._recipe_with(["lettuce"], time=45)
        self.assertNotIn("quick", generate_tags(r))

    def test_slow_cook_tag(self):
        r = self._recipe_with(["beef"], time=90)
        self.assertIn("slow-cook", generate_tags(r))

    def test_protein_rich_tag(self):
        r = self._recipe_with(["chicken"], time=25)
        self.assertIn("protein-rich", generate_tags(r))

    def test_baking_tag_from_ingredient(self):
        r = self._recipe_with(["flour", "baking soda"], time=45)
        self.assertIn("baking", generate_tags(r))

    def test_baking_tag_from_instruction(self):
        r = self._recipe_with(["oats"], instructions="Preheat oven to 350F.", time=30)
        self.assertIn("baking", generate_tags(r))

    def test_vegan_friendly(self):
        r = self._recipe_with(["flour", "sugar", "oil"], time=20)
        self.assertIn("vegan-friendly", generate_tags(r))

    def test_not_vegan_friendly_with_butter(self):
        r = self._recipe_with(["butter", "sugar"], time=20)
        self.assertNotIn("vegan-friendly", generate_tags(r))

    def test_vegetarian(self):
        r = self._recipe_with(["cheese", "flour"], time=20)
        self.assertIn("vegetarian", generate_tags(r))

    def test_not_vegetarian_with_chicken(self):
        r = self._recipe_with(["chicken"], time=25)
        self.assertNotIn("vegetarian", generate_tags(r))

    def test_blended_tag_from_instruction(self):
        r = self._recipe_with(["banana", "milk"], instructions="Blend everything until smooth.", time=5)
        self.assertIn("blended", generate_tags(r))

    def test_blended_tag_from_name(self):
        recipe = {"name": "Quick Smoothie", "ingredients": [{"item": "banana"}],
                  "instructions": [], "prep_time_minutes": 5}
        self.assertIn("blended", generate_tags(recipe))

    def test_tags_sorted(self):
        r = self._recipe_with(["flour", "baking soda"], time=20)
        tags = generate_tags(r)
        self.assertEqual(tags, sorted(tags))

    def test_no_time_no_quick_slow(self):
        r = self._recipe_with(["lettuce"], time=None)
        tags = generate_tags(r)
        self.assertNotIn("quick", tags)
        self.assertNotIn("slow-cook", tags)


# ---------------------------------------------------------------------------
class TestContentHash(unittest.TestCase):

    def _recipe(self, name, items):
        return {
            "name": name,
            "ingredients": [{"item": i} for i in items],
        }

    def test_same_recipe_same_hash(self):
        r = self._recipe("Tacos", ["chicken", "salsa", "tortillas"])
        self.assertEqual(content_hash(r), content_hash(r))

    def test_different_name_different_hash(self):
        r1 = self._recipe("Tacos",   ["chicken"])
        r2 = self._recipe("Burrito", ["chicken"])
        self.assertNotEqual(content_hash(r1), content_hash(r2))

    def test_different_ingredients_different_hash(self):
        r1 = self._recipe("Dish", ["chicken"])
        r2 = self._recipe("Dish", ["beef"])
        self.assertNotEqual(content_hash(r1), content_hash(r2))

    def test_ingredient_order_irrelevant(self):
        # Hash uses sorted ingredients, so order shouldn't matter
        r1 = self._recipe("Dish", ["a", "b", "c"])
        r2 = self._recipe("Dish", ["c", "a", "b"])
        self.assertEqual(content_hash(r1), content_hash(r2))

    def test_case_insensitive(self):
        r1 = self._recipe("Tacos", ["Chicken"])
        r2 = self._recipe("tacos", ["chicken"])
        self.assertEqual(content_hash(r1), content_hash(r2))


# ---------------------------------------------------------------------------
class TestDetectDuplicates(unittest.TestCase):

    def _make(self, rid, name, items):
        return {
            "recipe_id": rid, "name": name,
            "ingredients": [{"item": i} for i in items],
            "_meta": {"is_duplicate": False, "duplicate_of": None,
                      "imputed_fields": [], "original_values": [], "parse_warnings": []},
        }

    def test_no_duplicates_unchanged(self):
        recipes = [
            self._make("r01", "Tacos",   ["chicken"]),
            self._make("r02", "Pancakes", ["flour"]),
        ]
        result = detect_duplicates(recipes)
        self.assertFalse(result[0]["_meta"]["is_duplicate"])
        self.assertFalse(result[1]["_meta"]["is_duplicate"])

    def test_exact_duplicate_flagged(self):
        recipes = [
            self._make("r01", "Tacos", ["chicken", "salsa"]),
            self._make("r02", "Tacos", ["chicken", "salsa"]),
        ]
        result = detect_duplicates(recipes)
        self.assertFalse(result[0]["_meta"]["is_duplicate"])
        self.assertTrue(result[1]["_meta"]["is_duplicate"])
        self.assertEqual(result[1]["_meta"]["duplicate_of"], "r01")

    def test_duplicate_of_points_to_first(self):
        recipes = [
            self._make("r01", "Soup", ["tomato"]),
            self._make("r02", "Soup", ["tomato"]),
            self._make("r03", "Soup", ["tomato"]),
        ]
        result = detect_duplicates(recipes)
        self.assertEqual(result[1]["_meta"]["duplicate_of"], "r01")
        self.assertEqual(result[2]["_meta"]["duplicate_of"], "r01")

    def test_different_order_same_hash(self):
        recipes = [
            self._make("r01", "Dish", ["a", "b"]),
            self._make("r02", "Dish", ["b", "a"]),
        ]
        result = detect_duplicates(recipes)
        # Same hash → second is a duplicate
        self.assertTrue(result[1]["_meta"]["is_duplicate"])


if __name__ == "__main__":
    unittest.main()
