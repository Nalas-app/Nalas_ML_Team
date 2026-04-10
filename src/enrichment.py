"""
enrichment.py
-------------
Stage 2 – Enrichment functions.
SI unit conversion, ingredient normalisation, auto-tagging, duplicate detection.

After parsing, each recipe is a raw structured dict. This stage adds derived
information that makes the dataset more useful for downstream analysis:
  - All quantities are expressed in consistent SI base units (ml, g)
  - Descriptive tags are auto-generated from ingredient content and instructions
  - Duplicate recipes are detected and flagged using a content fingerprint
"""

from __future__ import annotations
import hashlib
import logging
from typing import Any

from .config import (
    SI_CONVERSION, PROTEIN_KEYS, VEGAN_KEYS,
    VEG_KEYS, BAKING_KEYS,
)

log = logging.getLogger("recipe_pipeline.enrichment")


# ---------------------------------------------------------------------------
# SI Unit Conversion
# ---------------------------------------------------------------------------

def convert_ingredient_to_si(ing: dict[str, Any]) -> dict[str, Any]:
    """Add si_quantity and si_unit to an ingredient dict (non-destructive copy).

    Looks up the ingredient's canonical unit in SI_CONVERSION and multiplies
    the quantity by the conversion factor. If the unit is not in SI_CONVERSION
    (e.g. "pinch", "clove"), the ingredient is returned unchanged.

    A shallow copy is made so the original dict is never mutated in-place —
    this keeps the pipeline stages independent and safe to re-run.

    Args:
        ing: A single ingredient dict with at least 'unit' and 'quantity' keys.

    Returns:
        The same dict (or a copy) with 'si_quantity' and 'si_unit' added if applicable.
    """
    unit = ing.get("unit")
    qty  = ing.get("quantity")
    if unit and qty is not None and unit in SI_CONVERSION:
        si_unit, factor = SI_CONVERSION[unit]
        ing = dict(ing)                          # shallow copy to avoid mutating the original
        ing["si_quantity"] = round(qty * factor, 3)   # convert and round to 3 decimal places
        ing["si_unit"]     = si_unit
    return ing


def convert_recipe_to_si(recipe: dict[str, Any]) -> dict[str, Any]:
    """Apply SI conversion to every ingredient in a recipe.

    Iterates over the recipe's ingredient list and converts each one.
    The recipe dict is mutated in-place (ingredients list is replaced).

    Args:
        recipe: A full recipe dict as produced by the parser.

    Returns:
        The same recipe dict with SI fields added to each ingredient.
    """
    recipe["ingredients"] = [convert_ingredient_to_si(i) for i in recipe.get("ingredients", [])]
    return recipe


# ---------------------------------------------------------------------------
# Auto-tagging
# ---------------------------------------------------------------------------

def generate_tags(recipe: dict[str, Any]) -> list[str]:
    """Auto-generate descriptive tags from recipe content.

    Tags are derived from three sources:
      1. Prep time: "quick" (≤30 min) or "slow-cook" (>60 min)
      2. Ingredient lists: compared against ingredient key sets in config.py
      3. Instruction text: scanned for keywords like "bake", "blend"

    Tags produced (may overlap):
        quick          – prep time is 30 minutes or less
        slow-cook      – prep time exceeds 60 minutes
        protein-rich   – contains at least one protein source (meat, tofu, egg…)
        vegan-friendly – contains no animal products
        vegetarian     – contains no meat/fish (dairy/eggs still allowed)
        baking         – uses flour/baking powder or mentions oven/bake
        blended        – instructions mention blending or recipe name says "smoothie"

    Args:
        recipe: A fully parsed and SI-converted recipe dict.

    Returns:
        Sorted list of tag strings (sorted for deterministic output in tests).
    """
    tags: set[str] = set()

    # Build a lowercase set of all ingredient item names for quick set intersection
    items_lower  = {(ing.get("item") or "").lower() for ing in recipe.get("ingredients", [])}
    # Flatten all instruction steps into a single string for keyword search
    instrs_lower = " ".join(recipe.get("instructions", [])).lower()
    time         = recipe.get("prep_time_minutes")

    # Tag by prep time
    if time is not None and time <= 30:
        tags.add("quick")
    if time is not None and time > 60:
        tags.add("slow-cook")

    # Tag by protein content (intersection with PROTEIN_KEYS set)
    if items_lower & PROTEIN_KEYS:
        tags.add("protein-rich")

    # Tag vegan-friendly only if NO animal product ingredients are present
    if not (items_lower & VEGAN_KEYS):
        tags.add("vegan-friendly")

    # Tag vegetarian only if NO meat/fish ingredients are present
    if not (items_lower & VEG_KEYS):
        tags.add("vegetarian")

    # Tag baking if baking ingredients are used OR oven/bake appears in instructions
    if items_lower & BAKING_KEYS or "bake" in instrs_lower or "oven" in instrs_lower:
        tags.add("baking")

    # Tag blended if the word "blend" appears in instructions or name contains "smoothie"
    if "blend" in instrs_lower or "smoothie" in (recipe.get("name") or "").lower():
        tags.add("blended")

    return sorted(tags)   # sort for deterministic ordering across runs


# ---------------------------------------------------------------------------
# Duplicate Detection
# ---------------------------------------------------------------------------

def content_hash(recipe: dict[str, Any]) -> str:
    """Compute an MD5 fingerprint based on recipe name + sorted ingredient items.

    The hash is used to detect structurally identical recipes regardless of
    how they appear in the source data. Note: this is a content hash, not a
    cryptographic one — MD5 is chosen for speed, not security.

    Logic:
        key = lowercased recipe name
        ingr_str = sorted ingredient item names joined by '|'
        hash = MD5(key + '::' + ingr_str)

    Args:
        recipe: A parsed recipe dict.

    Returns:
        32-character hex MD5 digest string.
    """
    key      = (recipe.get("name") or "").lower().strip()
    ingr_str = "|".join(
        sorted((ing.get("item") or "").lower().strip() for ing in recipe.get("ingredients", []))
    )
    return hashlib.md5(f"{key}::{ingr_str}".encode()).hexdigest()


def detect_duplicates(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark recipes whose name+ingredient fingerprint matches an earlier record.

    Iterates through recipes in order. The first occurrence of a fingerprint
    is stored as the canonical version. Any subsequent recipe with the same
    fingerprint is flagged as a duplicate in its _meta block.

    This means the FIRST occurrence is kept as the primary record; all later
    occurrences are marked with is_duplicate=True and duplicate_of=<recipe_id>.

    Args:
        recipes: List of parsed (and SI-converted) recipe dicts.

    Returns:
        Same list with _meta.is_duplicate and _meta.duplicate_of set where applicable.
    """
    seen: dict[str, str] = {}   # maps content hash → recipe_id of the first occurrence
    for recipe in recipes:
        h   = content_hash(recipe)
        rid = recipe.get("recipe_id", "unknown")
        if h in seen:
            # This recipe is a duplicate of an earlier one
            recipe["_meta"]["is_duplicate"] = True
            recipe["_meta"]["duplicate_of"] = seen[h]
            log.warning(f"Duplicate: {rid} matches {seen[h]}")
        else:
            # First time we've seen this fingerprint — register it
            seen[h] = rid
    return recipes


# ---------------------------------------------------------------------------
# Combined enrichment pass
# ---------------------------------------------------------------------------

def enrich(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run all enrichment steps on the full recipe list.

    Order matters:
        1. SI conversion — must happen before tagging so quantity checks work
        2. Duplicate detection — uses ingredient items, not SI values
        3. Tag generation — uses both ingredients and instructions

    Args:
        recipes: List of parsed recipe dicts (output of Stage 1).

    Returns:
        Enriched list with SI fields, duplicate flags, and tags added.
    """
    recipes = [convert_recipe_to_si(r) for r in recipes]   # Step 1: SI conversion
    recipes = detect_duplicates(recipes)                     # Step 2: duplicate flagging
    for recipe in recipes:
        recipe["tags"] = generate_tags(recipe)              # Step 3: auto-tagging
    return recipes
