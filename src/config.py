"""
config.py
---------
All constants, lookup tables, and compiled regex patterns for the pipeline.
Import this module everywhere instead of duplicating magic values.
"""

import re

# ---------------------------------------------------------------------------
# Unit normalisation map  (alias → canonical)
# Maps various spellings and plural forms of measurement units to a single
# canonical form used consistently across the whole pipeline.
# ---------------------------------------------------------------------------
UNIT_MAP: dict[str, str] = {
    # Volume units — both plural and abbreviation forms
    "cups": "cup", "c": "cup",
    "tablespoons": "tbsp", "tablespoon": "tbsp",
    "teaspoons": "tsp", "teaspoon": "tsp",
    "fluid ounces": "fl oz", "fl. oz": "fl oz",
    "milliliters": "ml", "millilitres": "ml", "milliliter": "ml",
    "liters": "l", "litres": "l", "liter": "l",
    # Mass units — British and American spelling variants included
    "ounces": "oz", "ounce": "oz",
    "grams": "g", "gram": "g",
    "kilograms": "kg", "kilogram": "kg",
    "pounds": "lb", "pound": "lb",
    # Time units — used internally when parsing time strings from recipe text
    "mins": "minutes", "min": "minutes",
    "hrs": "hours", "hr": "hours", "hour": "hours",
    # Miscellaneous cooking-specific units
    "pinches": "pinch", "sticks": "stick",
    "cloves": "clove", "slices": "slice",
}

# ---------------------------------------------------------------------------
# SI conversion factors  (canonical unit → (si_unit, factor))
# Used in enrichment.py to convert recipe quantities into SI base units
# so that ingredients across different recipes are comparable.
# Volume is normalised to millilitres (ml); mass to grams (g).
# ---------------------------------------------------------------------------
SI_CONVERSION: dict[str, tuple[str, float]] = {
    # Volume → ml conversion factors
    "cup":   ("ml",  236.588),   # 1 US cup  = 236.588 ml
    "tbsp":  ("ml",   14.787),   # 1 tablespoon = 14.787 ml
    "tsp":   ("ml",    4.929),   # 1 teaspoon   = 4.929 ml
    "fl oz": ("ml",   29.574),   # 1 fluid oz   = 29.574 ml
    "l":     ("ml", 1000.0),     # 1 litre      = 1000 ml
    # Mass → g conversion factors
    "oz":    ("g",   28.350),    # 1 ounce     = 28.35 g
    "lb":    ("g",  453.592),    # 1 pound     = 453.592 g
    "kg":    ("g", 1000.0),      # 1 kilogram  = 1000 g
    # Pass-through: already in SI — factor is 1.0, unit stays the same
    "ml":    ("ml",    1.0),
    "g":     ("g",     1.0),
}

# ---------------------------------------------------------------------------
# Ingredient name normalisation  (alias → canonical)
# Collapses common ingredient variants to a single canonical name.
# This makes ingredient frequency counts and duplicate detection more accurate.
# For example, "granulated sugar", "white sugar", and "caster sugar" all map to "sugar".
# ---------------------------------------------------------------------------
INGREDIENT_NORM: dict[str, str] = {
    "granulated sugar": "sugar",
    "white sugar":      "sugar",
    "caster sugar":     "sugar",
    "powdered sugar":   "icing sugar",
    "all purpose flour":"flour",
    "all-purpose flour":"flour",
    "plain flour":      "flour",
    "whole milk":       "milk",
    "semi-skimmed milk":"milk",
    "unsalted butter":  "butter",
    "salted butter":    "butter",
    "sea salt":         "salt",
    "kosher salt":      "salt",
    "table salt":       "salt",
    "vegetable oil":    "oil",
    "olive oil":        "oil",
    "canola oil":       "oil",
    "black pepper":     "pepper",
    "ground black pepper": "pepper",
}

# ---------------------------------------------------------------------------
# Auto-tag ingredient sets
# These sets are used in enrichment.py to automatically generate descriptive
# tags for each recipe by checking which ingredients it contains.
# ---------------------------------------------------------------------------

# Ingredients that make a recipe "protein-rich"
PROTEIN_KEYS = {"chicken", "beef", "shrimp", "pork", "fish", "tofu", "egg", "tuna"}

# Ingredients that disqualify a recipe from being "vegan-friendly"
VEGAN_KEYS   = {"meat", "chicken", "beef", "pork", "shrimp", "milk", "butter", "egg", "cream", "cheese"}

# Ingredients that disqualify a recipe from being "vegetarian"
VEG_KEYS     = {"chicken", "beef", "pork", "shrimp", "fish", "tuna"}

# Ingredients associated with baking recipes
BAKING_KEYS  = {"flour", "baking soda", "baking powder", "yeast"}

# ---------------------------------------------------------------------------
# Section-header keywords
# Used in parser.py to detect when a line is a section heading rather than
# actual ingredient or instruction content.
# ---------------------------------------------------------------------------
INGREDIENT_HEADERS  = {"ingredient", "ingredients"}
INSTRUCTION_HEADERS = {
    "instruction", "instructions", "direction", "directions",
    "step", "steps", "method", "preparation",
}

# ---------------------------------------------------------------------------
# Outlier thresholds
# Any recipe whose numeric values exceed these bounds is flagged in the
# validation report. Values are intentionally generous to avoid false positives.
# ---------------------------------------------------------------------------
OUTLIER_TIME_MAX     = 600   # prep_time_minutes > 600 (10 hours) is suspicious
OUTLIER_SERVINGS_MAX = 100   # more than 100 servings is likely a data entry error
OUTLIER_SERVINGS_MIN = 1     # 0 or negative servings should never appear

# ---------------------------------------------------------------------------
# Compiled regex patterns
# Pre-compiled once at import time for performance across the full dataset.
# ---------------------------------------------------------------------------

# Matches an optional quantity number (integer, fraction, or range)
RX_QTY  = r"(?P<qty>\d+(?:[./]\d+)?(?:\s*[-\u2013]\s*\d+(?:[./]\d+)?)?)"

# Matches a unit of measurement — covers all canonical and common variant forms
RX_UNIT = (
    r"(?P<unit>cup|tbsp|tsp|fl\s*oz|ml|g|kg|oz|lb|pound|pinch|"
    r"tablespoon|teaspoon|gram|liter|litre|liters|litres|stick|clove|slice|ounce)s?"
)

# Full ingredient line pattern: optional qty + optional unit + required item name
RX_INGR  = rf"^[-\u2022*\s]*{RX_QTY}?\s*{RX_UNIT}?\s+(?P<item>.+)$"
INGR_PAT = re.compile(RX_INGR, re.IGNORECASE)

# Matches prep/cooking time with optional hours and minutes components
# Examples matched: "Prep time: 1 hr 30 min", "takes 45 minutes", "time: 2 hours"
RX_TIME = re.compile(
    r"(?:prep\s*time|cooking\s*time|time\s*takes?|time|takes?|prep)\s*:?\s*"
    r"(?:(?P<hours>\d+)\s*(?:hr|hour)s?\s*(?:and\s*)?)?(?P<mins>\d+)?\s*(?:min|minute)s?",
    re.IGNORECASE,
)

# Fallback pattern: hours-only time string, e.g. "time: 2 hours"
RX_TIME_H_ONLY = re.compile(
    r"(?:prep\s*time|cooking\s*time|time|takes?)\s*:?\s*(?P<hours>\d+)\s*(?:hr|hour)s?",
    re.IGNORECASE,
)

# Fallback pattern: minutes-only time string, e.g. "time: 20 min"
RX_TIME_MIN_ONLY = re.compile(
    r"(?:prep\s*time|cooking\s*time|time\s*takes?|time|takes?)\s*:?\s*(?P<mins>\d+)\s*(?:min|minute)s?",
    re.IGNORECASE,
)

# Matches servings/yield line, e.g. "Serves: 4", "yield: 6", "servings 2"
RX_SERVINGS = re.compile(
    r"(?:serving|yield|serve|serves|yields?)\s*:?\s*(?P<n>\d+)",
    re.IGNORECASE,
)

# Default imputation values (used in imputation.py when a field is missing)
DEFAULT_SERVINGS = 4   # Most common household portion size
