"""
validation.py
-------------
Stage 3 – Validation functions.
JSON-Schema checks, required-field checks, outlier detection, duplicate flagging.

This stage does NOT modify any recipe data — it only observes and reports.
All issues are collected into a flat list of report-row dicts that are later
exported as validation_report.csv.

Severity levels used:
    error   – critical data quality issue (missing required field, schema violation)
    warning – soft issue that likely needs attention (outlier, duplicate, missing optional)
    info    – low-priority note from earlier stages (parse warnings)
"""

from __future__ import annotations
import logging
from typing import Any

import jsonschema
from jsonschema import validate as jschema_validate

from .config import OUTLIER_TIME_MAX, OUTLIER_SERVINGS_MAX, OUTLIER_SERVINGS_MIN

log = logging.getLogger("recipe_pipeline.validation")


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def check_outliers(recipe: dict[str, Any]) -> list[str]:
    """Return a list of human-readable outlier descriptions.

    Checks numeric fields against the thresholds defined in config.py:
        - prep_time_minutes > OUTLIER_TIME_MAX (600 min = 10 hours)
        - servings > OUTLIER_SERVINGS_MAX (100)
        - servings < OUTLIER_SERVINGS_MIN (1)
        - ingredient count > 30 (unusual for a single recipe)

    Only fields that are actually present and of the correct type are checked —
    missing values (None) are skipped here and handled by the imputation stage.

    Args:
        recipe: A single recipe dict (post-enrichment).

    Returns:
        List of human-readable issue strings (empty if no outliers found).
    """
    issues: list[str] = []
    time = recipe.get("prep_time_minutes")
    svc  = recipe.get("servings")

    # Check prep time — very long cook times can indicate data entry errors
    if isinstance(time, (int, float)) and time > OUTLIER_TIME_MAX:
        issues.append(f"prep_time_minutes={time} exceeds threshold {OUTLIER_TIME_MAX}")

    # Check servings — both too-large and too-small values are suspicious
    if isinstance(svc, (int, float)):
        if svc > OUTLIER_SERVINGS_MAX:
            issues.append(f"servings={svc} exceeds max {OUTLIER_SERVINGS_MAX}")
        if svc < OUTLIER_SERVINGS_MIN:
            issues.append(f"servings={svc} below min {OUTLIER_SERVINGS_MIN}")

    # Check ingredient count — more than 30 ingredients is unusually complex
    if len(recipe.get("ingredients", [])) > 30:
        issues.append(f"ingredient count={len(recipe['ingredients'])} unusually high")

    return issues


# ---------------------------------------------------------------------------
# Full validation pass
# ---------------------------------------------------------------------------

def validate_all(recipes: list[dict[str, Any]], schema: dict) -> list[dict[str, Any]]:
    """
    Run all validation checks across every recipe.

    Each check produces one or more 'report row' dicts. All rows are accumulated
    into a flat list and returned. The list is later written to validation_report.csv.

    Check types performed (in order):
        schema           – JSON Schema Draft-07 violations
        missing_required – name / ingredients / instructions empty
        missing_optional – prep_time_minutes / servings absent
        duplicate        – content fingerprint matches another recipe
        outlier          – numeric value out of expected range
        parse_warning    – notes carried over from the parsing stage

    Args:
        recipes: Full list of enriched recipe dicts.
        schema:  Loaded JSON Schema dict (from recipe_schema.json).

    Returns:
        List of report-row dicts, each with keys:
            recipe_id, recipe_name, check_type, severity, field, description
    """
    report_rows: list[dict[str, Any]] = []

    for recipe in recipes:
        rid  = recipe.get("recipe_id", "unknown")
        name = recipe.get("name", "—")

        # ── JSON Schema validation ──────────────────────────────────────────
        # Uses jsonschema's Draft-07 validator. Each violation is reported
        # separately. Schema-level errors (bad schema file) are logged but
        # do not add a report row.
        try:
            jschema_validate(instance=recipe, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            report_rows.append({
                "recipe_id":   rid,
                "recipe_name": name,
                "check_type":  "schema",
                "severity":    "error",
                # e.path is a deque of keys/indices; join them to form a field path
                "field":       ".".join(str(p) for p in e.path) or "(root)",
                "description": e.message,
            })
        except jsonschema.exceptions.SchemaError as e:
            # The schema file itself is invalid — this is a developer error
            log.error(f"Invalid schema: {e.message}")

        # ── Required field checks ───────────────────────────────────────────
        # name, ingredients, and instructions must be non-empty.
        # Empty lists and None both fail the truthiness test here.
        for field in ("name", "ingredients", "instructions"):
            if not recipe.get(field):
                report_rows.append({
                    "recipe_id":   rid,
                    "recipe_name": name,
                    "check_type":  "missing_required",
                    "severity":    "error",
                    "field":       field,
                    "description": f"Required field '{field}' is empty.",
                })

        # ── Optional field checks ───────────────────────────────────────────
        # Missing prep_time_minutes or servings are warnings (not errors)
        # because imputation.py will fill them in during Stage 4.
        for field in ("prep_time_minutes", "servings"):
            if recipe.get(field) is None:
                report_rows.append({
                    "recipe_id":   rid,
                    "recipe_name": name,
                    "check_type":  "missing_optional",
                    "severity":    "warning",
                    "field":       field,
                    "description": f"'{field}' is missing (will be imputed).",
                })

        # ── Duplicate check ─────────────────────────────────────────────────
        # The is_duplicate flag was set during the enrichment stage.
        # Report it here so it appears in the validation CSV.
        if recipe.get("_meta", {}).get("is_duplicate"):
            dup_of = recipe["_meta"].get("duplicate_of", "unknown")
            report_rows.append({
                "recipe_id":   rid,
                "recipe_name": name,
                "check_type":  "duplicate",
                "severity":    "warning",
                "field":       "name + ingredients",
                "description": f"Duplicate of recipe {dup_of}.",
            })

        # ── Outlier checks ──────────────────────────────────────────────────
        # check_outliers() returns a list of strings; each becomes one row.
        # The field name is extracted from the message (everything before "=").
        for msg in check_outliers(recipe):
            report_rows.append({
                "recipe_id":   rid,
                "recipe_name": name,
                "check_type":  "outlier",
                "severity":    "warning",
                "field":       msg.split("=")[0],   # e.g. "prep_time_minutes"
                "description": msg,
            })

        # ── Parse warnings (from Stage 1) ───────────────────────────────────
        # The _meta.parse_warnings list was populated in parser.py.
        # Surface each one as an "info" row so they appear in the report.
        for warn in recipe.get("_meta", {}).get("parse_warnings", []):
            report_rows.append({
                "recipe_id":   rid,
                "recipe_name": name,
                "check_type":  "parse_warning",
                "severity":    "info",
                "field":       "—",   # parse warnings are not field-specific
                "description": warn,
            })

    return report_rows
