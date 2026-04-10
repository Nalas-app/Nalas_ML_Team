"""
imputation.py
-------------
Stage 4 – Missing value imputation.
Rule-based (servings) + sklearn-median (prep_time_minutes).
All imputed values are logged in the recipe's _meta block.

Two strategies are used deliberately:
  - Statistical (sklearn median) for prep_time_minutes:
      Median is preferred over mean because cook-time distributions are
      right-skewed (a few very long recipes pull the mean upward).
  - Rule-based default for servings:
      A fixed default of 4 is used because servings is a categorical
      human decision, not a measurable continuous variable.

Every imputed value is recorded in _meta so downstream consumers and the
imputation_log.json output can distinguish original from inferred values.
"""

from __future__ import annotations
import logging
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer

from .config import DEFAULT_SERVINGS

log = logging.getLogger("recipe_pipeline.imputation")


def impute_missing_values(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply two imputation strategies to fill missing numeric recipe fields.

    Strategy 1 — prep_time_minutes (sklearn SimpleImputer, median):
        - All known prep times are extracted into a numpy array.
        - sklearn's SimpleImputer is fitted on this array (NaN-safe).
        - Missing values are replaced with the dataset-level median.
        - If ALL values are missing (entirely unknown dataset), falls back to
          DEFAULT_FALLBACK_TIME = 30 minutes.

    Strategy 2 — servings (rule-based default):
        - Any missing servings is filled with DEFAULT_SERVINGS = 4.
        - This constant represents the most common household portion size.

    For every imputed recipe the ``_meta`` block is updated:
        - ``imputed_fields``  – list of field names that were filled in
        - ``original_values`` – dict of {field: None} capturing pre-imputation state

    Args:
        recipes: List of structured recipe dicts (post-enrichment).

    Returns:
        Same list with missing values filled in (mutated in-place).
    """
    # Extract prep_time_minutes from all recipes into a numpy float array.
    # None values become np.nan so sklearn can handle them.
    times = np.array([r.get("prep_time_minutes") for r in recipes], dtype=float)

    # Fit the median imputer on the full array (including NaNs — sklearn ignores them)
    time_imputer = SimpleImputer(strategy="median")
    times_imp    = time_imputer.fit_transform(times.reshape(-1, 1)).flatten()

    # Compute our own fallback median in case sklearn returns NaN
    # (this only happens if every single prep_time_minutes value is missing)
    DEFAULT_FALLBACK_TIME = 30                             # minutes — a reasonable default
    known_times  = times[~np.isnan(times)]                 # filter out NaN values
    median_time  = float(np.nanmedian(known_times)) if len(known_times) else DEFAULT_FALLBACK_TIME

    log.info(f"Median prep time (known): {median_time:.0f} min | Default servings: {DEFAULT_SERVINGS}")

    for i, recipe in enumerate(recipes):
        # Ensure _meta exists even if this recipe was constructed without it
        meta = recipe.setdefault("_meta", {
            "imputed_fields":  [],
            "original_values": {},
            "parse_warnings":  [],
            "is_duplicate":    False,
            "duplicate_of":    None,
        })

        # ── Prep-time imputation (statistical / ML) ─────────────────────────
        if recipe.get("prep_time_minutes") is None:
            # Get the sklearn-imputed value for this index
            raw_val = times_imp[i] if len(times_imp) > i else float('nan')
            # NaN check: raw_val != raw_val is True only for IEEE 754 NaN
            imputed = DEFAULT_FALLBACK_TIME if (raw_val != raw_val) else int(round(raw_val))

            # Record the original None in the audit trail before overwriting
            meta["original_values"]["prep_time_minutes"] = None
            recipe["prep_time_minutes"] = imputed
            meta["imputed_fields"].append("prep_time_minutes")
            log.info(f"  [{recipe['recipe_id']}] prep_time_minutes imputed = {imputed} min (median)")

        # ── Servings imputation (rule-based) ────────────────────────────────
        if recipe.get("servings") is None:
            # Record the original None in the audit trail before overwriting
            meta["original_values"]["servings"] = None
            recipe["servings"] = DEFAULT_SERVINGS          # typically 4
            meta["imputed_fields"].append("servings")
            log.info(f"  [{recipe['recipe_id']}] servings imputed = {DEFAULT_SERVINGS} (default)")

    return recipes


def build_imputation_log(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact audit log of every imputed recipe.

    Produces one entry per recipe that had at least one field imputed.
    Each entry captures:
        - recipe_id       — identifies the recipe
        - recipe_name     — human-readable name for the audit log
        - imputed_fields  — which fields were filled in
        - original_values — the original (None) values before imputation
        - final_values    — the values that were substituted in

    This log is written to imputation_log.json by the pipeline orchestrator.

    Args:
        recipes: List of post-imputation recipe dicts.

    Returns:
        List of log entry dicts (only recipes with imputed fields are included).
    """
    return [
        {
            "recipe_id":       r.get("recipe_id"),
            "recipe_name":     r.get("name"),
            "imputed_fields":  r["_meta"]["imputed_fields"],
            "original_values": r["_meta"]["original_values"],
            # Retrieve the final post-imputation value for each imputed field
            "final_values":    {f: r.get(f) for f in r["_meta"]["imputed_fields"]},
        }
        for r in recipes
        if r.get("_meta", {}).get("imputed_fields")   # skip recipes where nothing was imputed
    ]
