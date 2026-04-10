"""
analytics.py
------------
Stage 5 – Dataset-level statistics and pipeline summary export.

This is the final stage of the pipeline. It does not modify any recipe data;
it only reads the enriched, validated, and imputed recipes to compute aggregate
statistics and write a machine-readable summary JSON.

The statistics produced here are:
  - Overall counts (total recipes, duplicates, imputed values)
  - Missing-rate percentages for optional fields
  - Descriptive statistics (mean, median, std, min, max) for numeric fields
  - Average ingredient and instruction step counts per recipe
  - Top-15 most frequent ingredients across the whole dataset
  - Tag distribution showing how many recipes carry each auto-generated tag
"""

from __future__ import annotations
import json
import logging
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

log = logging.getLogger("recipe_pipeline.analytics")


def _numeric_stats(values: list[float]) -> dict[str, Any]:
    """Return count/mean/median/std/min/max for a numeric list.

    All values are rounded to 2 decimal places for readability.
    Returns a dict with all keys set to None if the input list is empty,
    so callers don't need to guard against empty datasets.

    Args:
        values: List of numeric values (already filtered to exclude None).

    Returns:
        Dict with keys: count, mean, median, std, min, max.
    """
    if not values:
        # Return a safe empty-stats dict rather than raising an error
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "count":  int(len(arr)),
        "mean":   round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std":    round(float(np.std(arr)), 2),   # population std (not sample)
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }


def analyse_dataset(recipes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute dataset-level statistics across all recipes.

    This function is designed to be called AFTER imputation, so all numeric
    fields should already be populated. Missing-rate percentages reflect how
    many recipes required imputation in Stage 4.

    Args:
        recipes: Full list of enriched, validated, and imputed recipe dicts.

    Returns:
        A JSON-serialisable dict with the following keys:
            total_recipes              – total number of recipes in the dataset
            duplicate_count            – number of recipes flagged as duplicates
            imputed_prep_time_count    – how many recipes had prep_time imputed
            imputed_servings_count     – how many recipes had servings imputed
            missing_rate_prep_time_pct – percentage of recipes that needed imputation
            missing_rate_servings_pct  – percentage of recipes that needed imputation
            prep_time_minutes_stats    – descriptive stats for prep time
            servings_stats             – descriptive stats for servings
            avg_ingredients_per_recipe – mean ingredient count across all recipes
            avg_steps_per_recipe       – mean instruction step count across all recipes
            top_15_ingredients         – list of {item, count} dicts, most common first
            tag_distribution           – dict mapping each tag to its recipe count
    """
    n        = len(recipes)

    # Collect known (non-None) values for numeric stats
    times    = [r["prep_time_minutes"] for r in recipes if r.get("prep_time_minutes") is not None]
    servings = [r["servings"]          for r in recipes if r.get("servings")          is not None]

    # ── Ingredient frequency analysis ─────────────────────────────────────
    # Use normalized_item if available (canonical name), otherwise fall back
    # to the raw item string. This gives more accurate frequency counts by
    # collapsing aliases (e.g. "sea salt" and "kosher salt" → "salt").
    all_items = [
        (ing.get("normalized_item") or ing.get("item") or "").lower().strip()
        for r in recipes
        for ing in r.get("ingredients", [])
        if (ing.get("normalized_item") or ing.get("item") or "").strip()   # skip empty strings
    ]
    # Counter.most_common(15) returns the 15 most frequent items in descending order
    top_ingredients = [
        {"item": item, "count": count}
        for item, count in Counter(all_items).most_common(15)
    ]

    # ── Tag distribution ──────────────────────────────────────────────────
    # Flatten all tag lists into a single Counter for frequency analysis
    all_tags = [tag for r in recipes for tag in r.get("tags", [])]
    tag_dist = dict(Counter(all_tags).most_common())   # ordered by frequency descending

    # ── Imputation counts (from _meta audit trail) ────────────────────────
    imp_time = sum(1 for r in recipes if "prep_time_minutes" in r.get("_meta", {}).get("imputed_fields", []))
    imp_svc  = sum(1 for r in recipes if "servings"          in r.get("_meta", {}).get("imputed_fields", []))
    dups     = sum(1 for r in recipes if  r.get("_meta", {}).get("is_duplicate"))

    return {
        "total_recipes":              n,
        "duplicate_count":            dups,
        "imputed_prep_time_count":    imp_time,
        "imputed_servings_count":     imp_svc,
        # Missing rates: proportion of recipes that required imputation (as %)
        "missing_rate_prep_time_pct": round(imp_time / n * 100, 1) if n else 0,
        "missing_rate_servings_pct":  round(imp_svc  / n * 100, 1) if n else 0,
        "prep_time_minutes_stats":    _numeric_stats(times),
        "servings_stats":             _numeric_stats(servings),
        # Average ingredient and step counts per recipe
        "avg_ingredients_per_recipe": round(float(np.mean([len(r.get("ingredients", [])) for r in recipes])), 2) if n else 0,
        "avg_steps_per_recipe":       round(float(np.mean([len(r.get("instructions",  [])) for r in recipes])), 2) if n else 0,
        "top_15_ingredients":         top_ingredients,
        "tag_distribution":           tag_dist,
    }


def export_pipeline_summary(
    recipes:    list[dict[str, Any]],
    report:     list[dict[str, Any]],
    stats:      dict[str, Any],
    start_ts:   datetime,
    path:       str = "pipeline_summary.json",
) -> None:
    """Write a machine-readable pipeline-run summary JSON.

    Captures everything needed to reproduce or audit a pipeline run:
      - Timestamp and elapsed runtime
      - Total recipe count
      - Breakdown of validation issues by severity (error/warning/info)
      - Full dataset statistics from analyse_dataset()

    This file is useful for CI checks (e.g. assert errors == 0) and for
    tracking dataset quality trends across multiple pipeline runs over time.

    Args:
        recipes:   Full list of post-imputation recipe dicts.
        report:    Flat list of validation report row dicts (from Stage 3).
        stats:     Dataset statistics dict (from analyse_dataset()).
        start_ts:  datetime when the pipeline started (for elapsed calculation).
        path:      Output file path (default: pipeline_summary.json).
    """
    elapsed = (datetime.now() - start_ts).total_seconds()   # wall-clock time in seconds
    summary = {
        "run_timestamp":   datetime.now().isoformat(),       # ISO-8601 for easy parsing
        "elapsed_seconds": round(elapsed, 3),
        "total_recipes":   len(recipes),
        "validation_issues": {
            # Count each severity category from the flat report rows list
            "errors":   sum(1 for r in report if r["severity"] == "error"),
            "warnings": sum(1 for r in report if r["severity"] == "warning"),
            "info":     sum(1 for r in report if r["severity"] == "info"),
        },
        "dataset_stats": stats,   # embed the full statistics block
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)   # pretty-printed, UTF-8 safe
    log.info(f"Pipeline summary → {path}")
