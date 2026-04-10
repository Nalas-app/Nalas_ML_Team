"""
pipeline.py  (Orchestrator)
===========================
Thin CLI runner that imports all logic from the src/ package.
Run directly:  python pipeline.py [--dry-run] [--stats]
Or use the individual Jupyter notebooks in notebooks/ for step-by-step execution.

This file contains NO business logic — it only wires together the five pipeline
stages in the correct order and handles:
  - CLI argument parsing (argparse)
  - File I/O (loading input JSON, writing all output files)
  - Logging configuration
  - Progress reporting and summary statistics
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from datetime import datetime

# Import each pipeline stage from the src package
from src.parser     import extract_recipe_features
from src.enrichment import enrich
from src.validation import validate_all
from src.imputation import impute_missing_values, build_imputation_log
from src.analytics  import analyse_dataset, export_pipeline_summary

import pandas as pd

# Configure root logger: timestamps + level + message, written to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("recipe_pipeline")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_json(data: object, path: str) -> None:
    """Write any JSON-serialisable object to a file with pretty-printing.

    Uses ensure_ascii=False so that non-ASCII characters (e.g. accented
    ingredient names) are written as-is rather than escaped as \\uXXXX.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_csv(rows: list[dict], path: str, columns: list[str]) -> None:
    """Write a list of dicts to a CSV file using the specified column order.

    If the rows list is empty (no validation issues), an empty file is created
    rather than raising a KeyError on column selection.

    Args:
        rows:    List of flat dicts to write.
        path:    Destination file path.
        columns: Ordered list of column names to include.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        open(path, "w").close()   # create an empty file rather than skipping
    else:
        df[columns].to_csv(path, index=False, encoding="utf-8")


def _print_stats(stats: dict) -> None:
    """Log a formatted summary of dataset statistics to the console.

    Displays the most important metrics from analyse_dataset() in a
    human-readable table format. Called only when --stats is passed.
    """
    t = stats["prep_time_minutes_stats"]   # shorthand for nested dict
    s = stats["servings_stats"]
    log.info("")
    log.info("  ── DATASET STATISTICS ───────────────────────────")
    log.info(f"  Total recipes    : {stats['total_recipes']}")
    log.info(f"  Duplicates       : {stats['duplicate_count']}")
    log.info(f"  Missing time     : {stats['missing_rate_prep_time_pct']}%")
    log.info(f"  Missing servings : {stats['missing_rate_servings_pct']}%")
    log.info(f"  Prep time (min)  → mean={t['mean']}, median={t['median']}, std={t['std']}, range=[{t['min']}, {t['max']}]")
    log.info(f"  Servings         → mean={s['mean']}, median={s['median']}, std={s['std']}, range=[{s['min']}, {s['max']}]")
    log.info(f"  Avg ingredients  : {stats['avg_ingredients_per_recipe']}")
    log.info(f"  Avg steps        : {stats['avg_steps_per_recipe']}")
    # Show only the top 5 most frequent ingredients inline
    top5 = ", ".join(f"{e['item']}({e['count']})" for e in stats["top_15_ingredients"][:5])
    log.info(f"  Top-5 ingredients: {top5}")
    # Show only the top 6 tags inline
    tags = ", ".join(f"{k}:{v}" for k, v in list(stats["tag_distribution"].items())[:6])
    log.info(f"  Tags             : {tags}")
    log.info("  ─────────────────────────────────────────────────")
    log.info("")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_file:    str  = "mock_raw_data.json",    # path to the raw input JSON
    schema_file: str  = "recipe_schema.json",    # path to the JSON Schema
    out_clean:   str  = "clean_recipes.json",    # output: enriched + imputed recipes
    out_report:  str  = "validation_report.csv", # output: flat validation issues
    out_implog:  str  = "imputation_log.json",   # output: imputed-value audit trail
    out_summary: str  = "pipeline_summary.json", # output: machine-readable run summary
    dry_run:     bool = False,                   # if True, skip all file writes
    show_stats:  bool = True,                    # if True, log dataset statistics
) -> None:
    """Execute all five pipeline stages end-to-end.

    Stages:
        0. Load      – read raw JSON input and JSON Schema from disk
        1. Parse     – convert raw text strings into structured recipe dicts
        2. Enrich    – SI conversion, duplicate detection, auto-tagging
        3. Validate  – schema checks, outlier detection, missing field reports
        4. Impute    – fill missing prep_time_minutes and servings
        5. Analyse   – compute dataset statistics, export all output files

    Args:
        raw_file:    Path to the raw recipe JSON file (list of {id, raw_text} dicts).
        schema_file: Path to the JSON Schema file used for validation.
        out_clean:   Destination for the final cleaned recipe dataset.
        out_report:  Destination for the validation issues CSV.
        out_implog:  Destination for the imputation audit log.
        out_summary: Destination for the pipeline run summary JSON.
        dry_run:     If True, all stages run normally but no files are written.
        show_stats:  If True, print dataset statistics to the log.
    """
    start_ts = datetime.now()   # record start time for elapsed calculation
    log.info("=" * 60)
    log.info("  RECIPE DATA PROCESSING PIPELINE  –  START")
    log.info("=" * 60)

    # ── Stage 0: Load ─────────────────────────────────────────────────────
    if not os.path.exists(raw_file):
        log.error(f"Input file not found: {raw_file}")
        return
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)   # list of {"id": ..., "raw_text": ...} dicts
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)      # JSON Schema dict for validation stage
    log.info(f"[1/5] Loaded {len(raw_items)} raw entries from '{raw_file}'")

    # ── Stage 1: Parse ────────────────────────────────────────────────────
    log.info("[2/5] Parsing …")
    recipes = []
    for item in raw_items:
        # Use the source ID if present; otherwise generate a padded zero-indexed ID
        rid    = item.get("id", f"rec_{len(recipes)+1:03d}")
        recipe = extract_recipe_features(item.get("raw_text", ""), recipe_id=rid)
        recipes.append(recipe)
        log.info(f"      [{rid}] '{recipe['name'][:40]}' – "
                 f"{len(recipe['ingredients'])} ing., {len(recipe['instructions'])} steps")

    # ── Stage 2: Enrich ───────────────────────────────────────────────────
    log.info("[3/5] Enriching (SI + tags + dedup) …")
    recipes = enrich(recipes)   # mutates recipes list in-place, returns same list

    # ── Stage 3: Validate (pre-imputation) ────────────────────────────────
    # Validation intentionally runs BEFORE imputation so that missing values
    # are flagged in the report as "will be imputed" rather than hidden.
    log.info("[4/5] Validating …")
    report_rows = validate_all(recipes, schema)
    errors   = sum(1 for r in report_rows if r["severity"] == "error")
    warnings = sum(1 for r in report_rows if r["severity"] == "warning")
    log.info(f"      {errors} errors, {warnings} warnings")

    # ── Stage 4: Impute ───────────────────────────────────────────────────
    log.info("[5/5] Imputing missing values …")
    recipes = impute_missing_values(recipes)          # fill missing fields
    imp_log = build_imputation_log(recipes)           # build audit trail

    # ── Stage 5: Analyse ─────────────────────────────────────────────────
    stats = analyse_dataset(recipes)
    if show_stats:
        _print_stats(stats)   # print human-readable summary to console

    # ── Save outputs ──────────────────────────────────────────────────────
    if not dry_run:
        log.info("Saving outputs …")

        # Clean dataset: all recipes with SI fields, tags, and imputed values
        _save_json(recipes, out_clean)
        log.info(f"  Clean dataset     → {out_clean}")

        # Validation CSV: one row per issue found across all recipes
        _save_csv(
            report_rows, out_report,
            ["recipe_id", "recipe_name", "check_type", "severity", "field", "description"],
        )
        log.info(f"  Validation report → {out_report} ({len(report_rows)} rows)")

        # Imputation log: one entry per recipe where a value was filled in
        _save_json(imp_log, out_implog)
        log.info(f"  Imputation log    → {out_implog} ({len(imp_log)} entries)")

        # Pipeline summary: machine-readable run metadata + stats
        export_pipeline_summary(recipes, report_rows, stats, start_ts, path=out_summary)
    else:
        log.info("[DRY-RUN] Skipping file writes.")

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_ts).total_seconds()
    dups    = stats["duplicate_count"]
    imputed = stats["imputed_servings_count"]

    log.info("=" * 60)
    log.info(f"  PIPELINE COMPLETE" + ("  [DRY-RUN]" if dry_run else ""))
    log.info(f"  Total recipes    : {len(recipes)}")
    log.info(f"  Duplicates found : {dups}")
    log.info(f"  Recipes imputed  : {imputed}")
    log.info(f"  Validation issues: {len(report_rows)}")
    log.info(f"  Elapsed time     : {elapsed:.2f}s")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser.

    All arguments have sensible defaults so the pipeline can be run with
    no arguments at all:  python pipeline.py
    The --stats and --no-stats flags are mutually exclusive overrides.
    """
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Recipe Data Processing Pipeline – parse, validate, impute, clean.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py\n"
            "  python pipeline.py --input data/recipes.json\n"
            "  python pipeline.py --dry-run --stats\n"
            "  python pipeline.py --no-stats --out-clean results.json\n\n"
            "For step-by-step execution, open notebooks/ in Jupyter."
        ),
    )
    # Input file paths
    p.add_argument("--input",      "-i", default="mock_raw_data.json",   metavar="FILE",
                   help="Path to raw input JSON (default: mock_raw_data.json)")
    p.add_argument("--schema",     "-s", default="recipe_schema.json",   metavar="FILE",
                   help="Path to JSON Schema file (default: recipe_schema.json)")

    # Output file paths — all have sensible defaults
    p.add_argument("--out-clean",        default="clean_recipes.json",   metavar="FILE",
                   help="Output path for cleaned recipes JSON")
    p.add_argument("--out-report",       default="validation_report.csv",metavar="FILE",
                   help="Output path for validation report CSV")
    p.add_argument("--out-implog",       default="imputation_log.json",  metavar="FILE",
                   help="Output path for imputation audit log JSON")
    p.add_argument("--out-summary",      default="pipeline_summary.json",metavar="FILE",
                   help="Output path for pipeline summary JSON")

    # Behaviour flags
    p.add_argument("--dry-run",    action="store_true", default=False,
                   help="Run all stages but skip writing files")
    p.add_argument("--stats",      action="store_true", default=False,
                   help="Print dataset statistics")
    p.add_argument("--no-stats",   action="store_true", default=False,
                   help="Suppress statistics output")
    p.add_argument("--verbose", "-v", action="store_true", default=False,
                   help="Enable DEBUG logging")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    # Upgrade root logger to DEBUG if --verbose is passed
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve --stats / --no-stats conflict:
    # If neither is passed, default to showing stats (show=True).
    # If --no-stats is passed, suppress stats regardless of --stats.
    show = not args.no_stats if (args.stats or args.no_stats) else True

    run_pipeline(
        raw_file    = args.input,
        schema_file = args.schema,
        out_clean   = args.out_clean,
        out_report  = args.out_report,
        out_implog  = args.out_implog,
        out_summary = args.out_summary,
        dry_run     = args.dry_run,
        show_stats  = show,
    )
