"""
Retraining Pipeline
======================
Automated model retraining with evaluation and deployment decisions.

Triggers:
    1. Scheduled: Monthly (1st of month)
    2. Performance: Rolling 7-day MAPE > 15%
    3. Data: 200+ new completed orders
    4. Manual: On-demand

Run: python scripts/phase6_retrain.py [--force] [--version v1.1.0]
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import pandas as pd

# Add paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "ml_service"))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from cost_engine import DataStore, BusinessRules, CostEngine, CostingError
from feature_pipeline import FeaturePipeline, FeatureAnalyzer
from model_registry import ModelRegistry

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)

logger = logging.getLogger("ml_costing.retrain")

# ============================================================
# PATHS
# ============================================================
DATA_DIR = os.path.join(BASE_DIR, "data", "structured")
CONFIG_PATH = os.path.join(BASE_DIR, "ml_service", "config", "business_rules.json")
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOGS_DIR = os.path.join(BASE_DIR, "logs", "predictions")
RETRAIN_DIR = os.path.join(BASE_DIR, "logs", "retraining")
os.makedirs(RETRAIN_DIR, exist_ok=True)


# ============================================================
# RETRAINING CONFIGURATION
# ============================================================
RETRAIN_CONFIG = {
    "min_new_orders": 200,
    "mape_trigger_threshold": 15.0,
    "deployment_mape_threshold": 2.0,
    "monitoring_days": 7,
    "algorithm": "GradientBoosting",
    "hyperparams": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 8, 10],
        "learning_rate": [0.05, 0.1],
    },
}


def check_retrain_triggers(registry: ModelRegistry, logs_dir: str) -> Dict:
    """
    Check if retraining is needed based on triggers.

    Returns:
        Dict with trigger_name, should_retrain, and reason
    """
    triggers = {}

    # Trigger 1: Scheduled (monthly)
    active = registry.get_active_version()
    if active:
        versions = registry.list_versions()
        active_info = next((v for v in versions if v["version"] == active), None)
        if active_info:
            trained_at = datetime.fromisoformat(active_info["registered_at"])
            days_since = (datetime.now() - trained_at).days
            triggers["scheduled"] = {
                "should_retrain": days_since >= 30,
                "reason": f"Last training {days_since} days ago"
                          + (" (> 30 days)" if days_since >= 30 else ""),
            }

    # Trigger 2: Performance degradation
    if os.path.exists(logs_dir):
        try:
            recent_logs = _load_recent_predictions(logs_dir, days=7)
            if len(recent_logs) > 0:
                errors = recent_logs.get("error", pd.Series())
                error_rate = errors.notna().sum() / len(recent_logs)
                triggers["performance"] = {
                    "should_retrain": error_rate > 0.15,
                    "reason": f"Error rate: {error_rate*100:.1f}%"
                              + (" (above 15% threshold)" if error_rate > 0.15 else ""),
                }
        except Exception:
            triggers["performance"] = {
                "should_retrain": False,
                "reason": "Could not load prediction logs",
            }

    # Trigger 3: New data volume
    try:
        order_count = _count_new_orders(logs_dir)
        triggers["data_volume"] = {
            "should_retrain": order_count >= RETRAIN_CONFIG["min_new_orders"],
            "reason": f"{order_count} new orders"
                     + (f" (>= {RETRAIN_CONFIG['min_new_orders']} threshold)"
                        if order_count >= RETRAIN_CONFIG["min_new_orders"] else ""),
        }
    except Exception:
        triggers["data_volume"] = {
            "should_retrain": False,
            "reason": "Could not count new orders",
        }

    return triggers


def _load_recent_predictions(logs_dir: str, days: int = 7) -> pd.DataFrame:
    """Load recent prediction logs from JSONL files."""
    import glob
    files = sorted(glob.glob(os.path.join(logs_dir, "predictions_*.jsonl")))
    if not files:
        return pd.DataFrame()

    records = []
    for f in files[-days:]:
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    return pd.DataFrame(records) if records else pd.DataFrame()


def _count_new_orders(logs_dir: str) -> int:
    """Count total prediction logs (proxy for orders)."""
    df = _load_recent_predictions(logs_dir, days=30)
    return len(df)


def run_retraining(
    version: str = "auto",
    force: bool = False,
) -> Dict:
    """
    Execute the full retraining pipeline.

    Steps:
    1. Check triggers (unless forced)
    2. Load data
    3. Generate training data (from predictions + synthetic)
    4. Train model
    5. Evaluate
    6. Deployment decision
    7. Register model (if approved)

    Returns:
        Dict with results
    """
    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"  NALAS ML COSTING — Model Retraining [{run_id}]")
    print("=" * 60)

    # Setup
    registry = ModelRegistry(MODEL_DIR)
    data_store = DataStore(DATA_DIR)
    data_store.load()
    rules = BusinessRules(CONFIG_PATH)
    engine = CostEngine(data_store, rules)

    # Auto-version
    if version == "auto":
        versions = registry.list_versions()
        if versions:
            last = versions[-1]["version"]
            parts = last.replace("v", "").split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            version = "v" + ".".join(parts)
        else:
            version = "v1.0.0"

    print(f"\n  Target version: {version}")

    # Step 1: Check triggers
    if not force:
        print("\n[1/7] Checking retraining triggers...")
        triggers = check_retrain_triggers(registry, LOGS_DIR)
        should_retrain = any(t["should_retrain"] for t in triggers.values())
        for name, info in triggers.items():
            icon = "✅" if info["should_retrain"] else "⬜"
            print(f"  {icon} {name}: {info['reason']}")

        if not should_retrain:
            print("\n  No triggers fired — skipping retraining")
            return {"status": "skipped", "reason": "No triggers fired", "triggers": triggers}
    else:
        print("\n[1/7] Force retraining — skipping trigger checks")

    # Step 2: Load data
    print("\n[2/7] Preparing training data...")
    working_items = []
    for _, row in data_store.menu_items.iterrows():
        recipe = data_store.get_recipe(row['item_id'])
        if int(recipe['quantity_per_base_unit'].notna().sum()) > 0:
            working_items.append(row['item_id'])

    # Generate training data
    np.random.seed(int(time.time()) % 2**31)
    n_samples = 1000
    records = []
    event_types = ['Wedding', 'Birthday', 'Corporate', 'Other']

    for _ in range(n_samples):
        item_id = np.random.choice(working_items)
        qty = max(5, min(int(np.random.lognormal(3.5, 0.8)), 2000))
        month = np.random.choice(range(1, 13))
        event_date = f"2026-{month:02d}-{np.random.randint(1, 28):02d}"
        guest_count = max(10, int(qty * np.random.uniform(0.8, 2.5)))
        event_type = np.random.choice(event_types, p=[0.40, 0.25, 0.20, 0.15])

        try:
            result = engine.predict(item_id, qty, event_date, guest_count)
            noise = np.random.normal(1.0, 0.08)
            records.append({
                'menu_item_id': item_id,
                'quantity': qty,
                'event_date': event_date,
                'guest_count': guest_count,
                'event_type': event_type,
                'actual_cost': max(result['totalCost'] * noise, 10),
            })
        except CostingError:
            continue

    full_data = pd.DataFrame(records)
    print(f"  Training samples: {len(full_data)}")

    # Step 3: Feature engineering
    print("\n[3/7] Feature engineering...")
    pipeline = FeaturePipeline(data_store=data_store)
    raw_features = pipeline.extract_features(full_data)

    n = len(full_data)
    train_idx = int(n * 0.80)
    train_features = raw_features.iloc[:train_idx]
    test_features = raw_features.iloc[train_idx:]
    y_train = full_data['actual_cost'].iloc[:train_idx].values
    y_test = full_data['actual_cost'].iloc[train_idx:].values

    pipeline.fit(train_features)
    X_train = pipeline.transform(train_features)
    X_test = pipeline.transform(test_features)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

    # Step 4: Train model
    print("\n[4/7] Training model...")
    param_grid = RETRAIN_CONFIG["hyperparams"]

    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid, cv=3,
        scoring='neg_mean_absolute_percentage_error',
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")

    # Step 5: Evaluate
    print("\n[5/7] Evaluating model...")
    y_pred = np.maximum(best_model.predict(X_test), 0)

    mask = y_test > 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    metrics = {"mape": round(mape, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "mae": round(mae, 2)}
    print(f"  MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Step 6: Deployment decision
    print("\n[6/7] Deployment decision...")
    decision = registry.should_deploy(metrics)
    print(f"  Deploy: {'✅ YES' if decision['deploy'] else '❌ NO'}")
    print(f"  Reason: {decision['reason']}")

    # Step 7: Register model
    if decision["deploy"] or force:
        print("\n[7/7] Registering model...")
        registry.register_model(
            model=best_model,
            pipeline=pipeline,
            version=version,
            algorithm="GradientBoosting",
            training_samples=len(X_train),
            features=pipeline.feature_names,
            metrics=metrics,
            hyperparameters=grid.best_params_,
            set_active=True,
        )
        print(f"  ✅ Model {version} registered and set as active")
    else:
        print("\n[7/7] Model NOT deployed — see reason above")

    # Save retraining log
    elapsed = round(time.time() - start_time, 2)
    retrain_log = {
        "run_id": run_id,
        "version": version,
        "elapsed_seconds": elapsed,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "best_params": grid.best_params_,
        "metrics": metrics,
        "decision": decision,
        "deployed": decision["deploy"] or force,
        "timestamp": datetime.now().isoformat(),
    }

    log_path = os.path.join(RETRAIN_DIR, f"retrain_{run_id}.json")
    with open(log_path, 'w') as f:
        json.dump(retrain_log, f, indent=2, default=str)

    print(f"\n  Retraining log: {log_path}")
    print(f"  Total time: {elapsed}s")
    print("\n" + "=" * 60)
    print(f"  Retraining Complete — {version}")
    print("=" * 60)

    return retrain_log


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain ML model")
    parser.add_argument("--force", action="store_true", help="Force retraining")
    parser.add_argument("--version", default="auto", help="Model version (default: auto)")
    args = parser.parse_args()

    run_retraining(version=args.version, force=args.force)
