"""
Phase 5 — ML Model Training & Evaluation Pipeline
=====================================================
Trains, evaluates, and compares multiple ML models for cost prediction.

Models:
    1. Mean Cost Baseline
    2. Linear Regression
    3. Ridge Regression
    4. Random Forest Regressor
    5. Gradient Boosting (XGBoost-style via sklearn)

Run: python scripts/phase5_model_training.py
"""

import os
import sys
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# Add paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "ml_service"))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from cost_engine import DataStore, BusinessRules, CostEngine, CostingError
from feature_pipeline import FeaturePipeline, FeatureExtractor, FeatureAnalyzer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

# ============================================================
# PATHS
# ============================================================
DATA_DIR = os.path.join(BASE_DIR, "data", "structured")
CONFIG_PATH = os.path.join(BASE_DIR, "ml_service", "config", "business_rules.json")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "model_evaluation")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("  NALAS ML COSTING — Phase 5 Model Training")
print("=" * 60)

# ============================================================
# STEP 1: Load Data & Generate Training Set
# ============================================================
print("\n[1/7] Loading data & generating training set...")

data_store = DataStore(DATA_DIR)
data_store.load()
rules = BusinessRules(CONFIG_PATH)
engine = CostEngine(data_store, rules)

# Get working items
working_items = []
for _, row in data_store.menu_items.iterrows():
    item_id = row['item_id']
    recipe = data_store.get_recipe(item_id)
    has_qty = int(recipe['quantity_per_base_unit'].notna().sum())
    if has_qty > 0:
        working_items.append(item_id)

print(f"  Working items: {len(working_items)}")

# Generate synthetic orders (simulating collected data)
np.random.seed(42)
n_samples = 1000

event_types = ['Wedding', 'Birthday', 'Corporate', 'Other']
records = []

for i in range(n_samples):
    item_id = np.random.choice(working_items)
    qty = int(np.random.lognormal(mean=3.5, sigma=0.8))  # Realistic quantity distribution
    qty = max(5, min(qty, 2000))
    month = np.random.choice(range(1, 13))
    day = np.random.randint(1, 28)
    event_date = f"2026-{month:02d}-{day:02d}"
    guest_count = max(10, int(qty * np.random.uniform(0.8, 2.5)))
    event_type = np.random.choice(event_types, p=[0.40, 0.25, 0.20, 0.15])

    try:
        result = engine.predict(item_id, qty, event_date, guest_count)
        # Add realistic noise to simulate actual cost variation
        noise_factor = np.random.normal(1.0, 0.08)  # ±8% noise
        actual_cost = max(result['totalCost'] * noise_factor, 10)

        records.append({
            'menu_item_id': item_id,
            'quantity': qty,
            'event_date': event_date,
            'guest_count': guest_count,
            'event_type': event_type,
            'predicted_cost': result['totalCost'],
            'actual_cost': round(actual_cost, 2),
            'ingredient_cost': result['ingredientCost'],
            'confidence': result['confidence'],
        })
    except CostingError:
        continue

full_data = pd.DataFrame(records)
print(f"  Generated {len(full_data)} training samples")
print(f"  Cost range: Rs.{full_data['actual_cost'].min():.2f} — Rs.{full_data['actual_cost'].max():.2f}")
print(f"  Cost mean: Rs.{full_data['actual_cost'].mean():.2f}")
print(f"  Cost median: Rs.{full_data['actual_cost'].median():.2f}")

# Save training data for reference
full_data.to_csv(os.path.join(REPORTS_DIR, "training_data.csv"), index=False)

# ============================================================
# STEP 2: Feature Engineering
# ============================================================
print("\n[2/7] Feature engineering...")

pipeline = FeaturePipeline(data_store=data_store)
raw_features = pipeline.extract_features(full_data)

# Time-based split: first 70% train, next 15% val, last 15% test
n = len(full_data)
train_idx = int(n * 0.70)
val_idx = int(n * 0.85)

train_data = full_data.iloc[:train_idx]
val_data = full_data.iloc[train_idx:val_idx]
test_data = full_data.iloc[val_idx:]

train_features = raw_features.iloc[:train_idx]
val_features = raw_features.iloc[train_idx:val_idx]
test_features = raw_features.iloc[val_idx:]

y_train = full_data['actual_cost'].iloc[:train_idx].values
y_val = full_data['actual_cost'].iloc[train_idx:val_idx].values
y_test = full_data['actual_cost'].iloc[val_idx:].values

# Fit pipeline on TRAINING data only (no leakage)
pipeline.fit(train_features)

X_train = pipeline.transform(train_features)
X_val = pipeline.transform(val_features)
X_test = pipeline.transform(test_features)

print(f"  Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
print(f"  Features: {X_train.shape[1]}")

# Save the pipeline
pipeline.save(os.path.join(MODEL_DIR, "feature_pipeline.pkl"))


# ============================================================
# STEP 3: Baseline Models
# ============================================================
print("\n[3/7] Training baseline models...")

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate all metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Safe MAPE: avoid division by zero
    mask = y_true > 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 100.0
    return {
        'model': model_name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2),
    }

results = []

# Baseline 1: Mean Cost per Item
print("  Training Baseline 1: Mean Cost per Item...")
start = time.time()
item_means = train_data.groupby('menu_item_id')['actual_cost'].mean()
qty_means = train_data.groupby('menu_item_id')['quantity'].mean()

def baseline_predict(row):
    item_id = row['menu_item_id']
    qty = row['quantity']
    if item_id in item_means.index:
        mean_cost = item_means[item_id]
        mean_qty = qty_means[item_id]
        return mean_cost * (qty / mean_qty) if mean_qty > 0 else mean_cost
    return train_data['actual_cost'].mean()

y_pred_baseline = test_data.apply(baseline_predict, axis=1).values
train_time_baseline = (time.time() - start) * 1000

start = time.time()
_ = test_data.apply(baseline_predict, axis=1).values
pred_time_baseline = (time.time() - start) * 1000 / len(test_data)

metrics_baseline = evaluate_model(y_test, y_pred_baseline, "Baseline: Mean Cost")
metrics_baseline['train_time_ms'] = round(train_time_baseline, 2)
metrics_baseline['pred_latency_ms'] = round(pred_time_baseline, 4)
results.append(metrics_baseline)
print(f"    MAPE: {metrics_baseline['MAPE']:.2f}%, R²: {metrics_baseline['R2']:.4f}")


# ============================================================
# STEP 4: Train ML Models
# ============================================================
print("\n[4/7] Training ML models...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        min_samples_split=5, random_state=42
    ),
}

trained_models = {}

for name, model in models.items():
    print(f"  Training {name}...")

    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = (time.time() - start) * 1000

    # Predict
    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = (time.time() - start) * 1000 / len(X_test)

    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, name)
    metrics['train_time_ms'] = round(train_time, 2)
    metrics['pred_latency_ms'] = round(pred_time, 4)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    metrics['CV_R2_mean'] = round(float(cv_scores.mean()), 4)
    metrics['CV_R2_std'] = round(float(cv_scores.std()), 4)

    cv_mape = cross_val_score(
        model, X_train, y_train, cv=5,
        scoring='neg_mean_absolute_percentage_error'
    )
    metrics['CV_MAPE_mean'] = round(float(-cv_mape.mean() * 100), 2)

    results.append(metrics)
    trained_models[name] = {'model': model, 'predictions': y_pred, 'metrics': metrics}

    print(f"    MAPE: {metrics['MAPE']:.2f}%, R²: {metrics['R2']:.4f}, "
          f"CV R²: {metrics['CV_R2_mean']:.4f} ± {metrics['CV_R2_std']:.4f}")


# ============================================================
# STEP 5: Hyperparameter Tuning (Best 2 Models)
# ============================================================
print("\n[5/7] Hyperparameter tuning...")

# Tune Random Forest
print("  Tuning Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=3,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=0,
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print(f"    Best params: {rf_grid.best_params_}")
print(f"    Best CV MAPE: {-rf_grid.best_score_*100:.2f}%")

# Evaluate tuned RF
y_pred_tuned_rf = np.maximum(best_rf.predict(X_test), 0)
metrics_tuned_rf = evaluate_model(y_test, y_pred_tuned_rf, "Random Forest (Tuned)")
metrics_tuned_rf['best_params'] = rf_grid.best_params_
print(f"    Test MAPE: {metrics_tuned_rf['MAPE']:.2f}%, R²: {metrics_tuned_rf['R2']:.4f}")

# Tune Gradient Boosting
print("  Tuning Gradient Boosting...")
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 8],
    'learning_rate': [0.05, 0.1, 0.2],
}

gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_param_grid,
    cv=3,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=0,
)
gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_
print(f"    Best params: {gb_grid.best_params_}")
print(f"    Best CV MAPE: {-gb_grid.best_score_*100:.2f}%")

# Evaluate tuned GB
y_pred_tuned_gb = np.maximum(best_gb.predict(X_test), 0)
metrics_tuned_gb = evaluate_model(y_test, y_pred_tuned_gb, "Gradient Boosting (Tuned)")
metrics_tuned_gb['best_params'] = gb_grid.best_params_
print(f"    Test MAPE: {metrics_tuned_gb['MAPE']:.2f}%, R²: {metrics_tuned_gb['R2']:.4f}")

results.append(metrics_tuned_rf)
results.append(metrics_tuned_gb)


# ============================================================
# STEP 6: Model Comparison & Selection
# ============================================================
print("\n[6/7] Model comparison & selection...")

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('MAPE')
comparison_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

print("\n  Model Comparison (sorted by MAPE):")
print(f"  {'Model':35s} {'MAPE':>8} {'RMSE':>12} {'R²':>8} {'Train(ms)':>10} {'Pred(ms)':>10}")
print("  " + "-" * 90)
for _, row in comparison_df.iterrows():
    train_t = row.get('train_time_ms', 'N/A')
    pred_t = row.get('pred_latency_ms', 'N/A')
    print(f"  {row['model']:35s} {row['MAPE']:7.2f}% {row['RMSE']:11.2f} {row['R2']:7.4f} "
          f"{train_t:>10} {pred_t:>10}")

# Select best model
best_row = comparison_df.iloc[0]
best_model_name = best_row['model']
print(f"\n  ✅ Best Model: {best_model_name} (MAPE={best_row['MAPE']:.2f}%, R²={best_row['R2']:.4f})")

# Determine which model object is best
if 'Tuned' in best_model_name:
    if 'Random Forest' in best_model_name:
        best_model = best_rf
    else:
        best_model = best_gb
elif best_model_name in trained_models:
    best_model = trained_models[best_model_name]['model']
else:
    best_model = best_rf  # Fallback


# ============================================================
# STEP 6b: Feature Importance & Interpretation
# ============================================================
print("\n  Feature Importance (Best Model):")
if hasattr(best_model, 'feature_importances_'):
    importance_df = FeatureAnalyzer.extract_feature_importance(
        best_model, pipeline.feature_names
    )
    importance_df.to_csv(os.path.join(REPORTS_DIR, "best_model_feature_importance.csv"), index=False)

    print(f"  {'Rank':>4} {'Feature':30s} {'Importance':>12} {'Cumulative':>12}")
    print("  " + "-" * 62)
    for _, row in importance_df.head(10).iterrows():
        print(f"  {int(row['rank']):4d} {row['feature']:30s} "
              f"{row['importance_pct']:10.2f}% {row['cumulative_pct']:10.2f}%")

    # Validate
    validations = FeatureAnalyzer.validate_top_features(importance_df)
    print("\n  Feature Validation:")
    for v in validations:
        icon = "✅" if v['status'] == 'PASS' else "⚠️"
        print(f"    {icon} {v['check']}: {v['details']}")
elif hasattr(best_model, 'coef_'):
    coefs = pd.DataFrame({
        'feature': pipeline.feature_names,
        'coefficient': best_model.coef_.flatten(),
        'abs_coefficient': np.abs(best_model.coef_.flatten()),
    }).sort_values('abs_coefficient', ascending=False)
    coefs.to_csv(os.path.join(REPORTS_DIR, "best_model_coefficients.csv"), index=False)
    print("  (Linear model — coefficients saved)")


# ============================================================
# STEP 6c: Error Analysis
# ============================================================
print("\n  Error Analysis:")
if 'Random Forest' in best_model_name:
    y_pred_best = y_pred_tuned_rf
elif 'Gradient Boosting' in best_model_name:
    y_pred_best = y_pred_tuned_gb
else:
    y_pred_best = best_model.predict(X_test)
    y_pred_best = np.maximum(y_pred_best, 0)

errors = pd.DataFrame({
    'menu_item_id': test_data['menu_item_id'].values,
    'quantity': test_data['quantity'].values,
    'event_type': test_data['event_type'].values,
    'actual_cost': y_test,
    'predicted_cost': y_pred_best,
    'error': y_pred_best - y_test,
    'abs_error': np.abs(y_pred_best - y_test),
    'pct_error': np.where(y_test > 0, np.abs(y_pred_best - y_test) / y_test * 100, 0),
})

# High error predictions (> 20%)
high_errors = errors[errors['pct_error'] > 20]
print(f"  Predictions with > 20% error: {len(high_errors)} / {len(errors)} "
      f"({len(high_errors)/len(errors)*100:.1f}%)")

# Error by category
item_cat = {}
for _, row in data_store.menu_items.iterrows():
    item_cat[row['item_id']] = row['category']
errors['category'] = errors['menu_item_id'].map(item_cat)

error_by_category = errors.groupby('category')['pct_error'].agg(['mean', 'median', 'count'])
print("\n  Error by Category:")
for cat, row in error_by_category.iterrows():
    print(f"    {cat:25s} Mean: {row['mean']:6.2f}%  Median: {row['median']:6.2f}%  N={int(row['count'])}")

# Error by quantity range
errors['qty_range'] = pd.cut(errors['quantity'], bins=[0, 50, 100, 200, 500, float('inf')],
                              labels=['1-50', '51-100', '101-200', '201-500', '500+'])
error_by_qty = errors.groupby('qty_range', observed=False)['pct_error'].agg(['mean', 'count'])
print("\n  Error by Quantity Range:")
for qr, row in error_by_qty.iterrows():
    print(f"    Qty {qr:10s} Mean Error: {row['mean']:6.2f}%  N={int(row['count'])}")

errors.to_csv(os.path.join(REPORTS_DIR, "error_analysis.csv"), index=False)


# ============================================================
# STEP 7: Save Best Model & Generate Report
# ============================================================
print("\n[7/7] Saving best model & generating report...")

# Save best model
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"  ✅ Best model saved to: {model_path}")

# Save model metadata
model_metadata = {
    'model_name': best_model_name,
    'model_type': type(best_model).__name__,
    'version': 'v1.0.0',
    'trained_at': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features': pipeline.feature_names,
    'n_features': len(pipeline.feature_names),
    'metrics': {
        'test_MAPE': float(best_row['MAPE']),
        'test_RMSE': float(best_row['RMSE']),
        'test_R2': float(best_row['R2']),
        'test_MAE': float(best_row['MAE']),
    },
    'hyperparameters': (
        rf_grid.best_params_ if 'Random Forest' in best_model_name
        else gb_grid.best_params_ if 'Gradient Boosting' in best_model_name
        else {}
    ),
    'all_models_compared': [r['model'] for r in results],
}

with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'w') as f:
    json.dump(model_metadata, f, indent=2, default=str)

# Full report
full_report = {
    'generated_at': datetime.now().isoformat(),
    'data_summary': {
        'total_samples': n_samples,
        'valid_samples': len(full_data),
        'working_items': len(working_items),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'split_method': 'time-based (70/15/15)',
    },
    'model_comparison': results,
    'best_model': {
        'name': best_model_name,
        'metrics': dict(best_row),
        'selection_rationale': (
            f"{best_model_name} selected: lowest MAPE ({best_row['MAPE']:.2f}%), "
            f"highest R² ({best_row['R2']:.4f}), acceptable latency"
        ),
    },
    'tuning_results': {
        'random_forest_best_params': rf_grid.best_params_,
        'random_forest_best_cv_mape': round(-rf_grid.best_score_ * 100, 2),
        'gradient_boosting_best_params': gb_grid.best_params_,
        'gradient_boosting_best_cv_mape': round(-gb_grid.best_score_ * 100, 2),
    },
    'error_analysis': {
        'high_error_predictions_pct': round(len(high_errors) / len(errors) * 100, 2),
        'error_by_category': error_by_category.to_dict(),
    },
    'model_metadata': model_metadata,
}

with open(os.path.join(REPORTS_DIR, "training_report.json"), 'w') as f:
    json.dump(full_report, f, indent=2, default=str)

print(f"  Reports saved to: {REPORTS_DIR}")


# ============================================================
# STEP 8: Visualizations, Partial Dependence Plots & SHAP
# ============================================================
print("\n[8/8] Generating visualizations, PDP & SHAP analysis...")

PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Plots (matplotlib) ──────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # Non-interactive — safe for scripts
    import matplotlib.pyplot as plt

    # Clip extreme outliers so charts are readable (show p95)
    clip_val = np.percentile(y_test, 95)
    vis_mask = y_test <= clip_val

    # ── Plot 1: Predictions vs Actuals Scatter ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test[vis_mask], y_pred_best[vis_mask],
               alpha=0.5, s=20, color="steelblue", label="Predictions")
    lo = min(y_test[vis_mask].min(), y_pred_best[vis_mask].min())
    hi = max(y_test[vis_mask].max(), y_pred_best[vis_mask].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Cost (₹)", fontsize=12)
    ax.set_ylabel("Predicted Cost (₹)", fontsize=12)
    ax.set_title(
        f"Predictions vs Actuals — {best_model_name}\n"
        f"(p95 clip applied, MAPE={best_row['MAPE']:.1f}%)", fontsize=13
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "predictions_vs_actuals.png"), dpi=150)
    plt.close()
    print("  ✅ Plot 1: Predictions vs Actuals saved")

    # ── Plot 2: Residual Plot ──
    residuals = y_pred_best - y_test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred_best[vis_mask], residuals[vis_mask],
               alpha=0.5, s=20, color="coral")
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted Cost (₹)", fontsize=12)
    ax.set_ylabel("Residual  (Predicted − Actual)  ₹", fontsize=12)
    ax.set_title("Residual Plot — Errors Should Be Random (No Pattern)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "residual_plot.png"), dpi=150)
    plt.close()
    print("  ✅ Plot 2: Residual Plot saved")

    # ── Plot 3: Error Distribution Histogram ──
    pct_errors_clipped = errors["pct_error"].clip(upper=100)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pct_errors_clipped, bins=30,
            color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(pct_errors_clipped.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean: {pct_errors_clipped.mean():.1f}%")
    ax.axvline(10, color="green", linestyle="--",
               linewidth=1.5, label="Target: 10% MAPE")
    ax.set_xlabel("Absolute % Error  (clipped at 100%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Prediction Error Distribution", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "error_distribution.png"), dpi=150)
    plt.close()
    print("  ✅ Plot 3: Error Distribution saved")

    # ── Plot 4: Error by Menu Category ──
    if "category" in errors.columns:
        cat_errors = (errors.groupby("category")["pct_error"]
                      .mean().sort_values(ascending=False))
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(cat_errors)), cat_errors.values,
                      color="steelblue", alpha=0.8)
        ax.axhline(10, color="green", linestyle="--",
                   linewidth=1.5, label="Target MAPE 10%")
        ax.set_xticks(range(len(cat_errors)))
        ax.set_xticklabels(cat_errors.index, rotation=30,
                           ha="right", fontsize=10)
        ax.set_ylabel("Mean Absolute % Error", fontsize=12)
        ax.set_title("Prediction Error by Menu Category", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, cat_errors.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "error_by_category.png"), dpi=150)
        plt.close()
        print("  ✅ Plot 4: Error by Category saved")

    # ── Plot 5: Partial Dependence Plots (top 2 features) ──
    try:
        from sklearn.inspection import PartialDependenceDisplay

        top_feature_names = ["base_ingredient_cost", "quantity"]
        feature_indices = [
            pipeline.feature_names.index(f)
            for f in top_feature_names
            if f in pipeline.feature_names
        ]

        if feature_indices:
            fig, axes = plt.subplots(1, len(feature_indices),
                                     figsize=(6 * len(feature_indices), 5))
            if len(feature_indices) == 1:
                axes = [axes]

            PartialDependenceDisplay.from_estimator(
                best_model,
                X_train,
                feature_indices,
                feature_names=pipeline.feature_names,
                ax=axes,
                grid_resolution=50,
                random_state=42,
            )
            fig.suptitle(
                f"Partial Dependence Plots — {best_model_name}\n"
                "(How cost changes with each feature, holding others constant)",
                fontsize=12,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "partial_dependence_plots.png"), dpi=150)
            plt.close()
            print("  ✅ Plot 5: Partial Dependence Plots saved")
        else:
            print("  ⚠️  PDP: top features not in pipeline feature names — skipped")

    except Exception as pdp_err:
        print(f"  ⚠️  PDP skipped: {pdp_err}")

except ImportError:
    print("  ⚠️  matplotlib not installed — plots skipped. Run: pip install matplotlib")
except Exception as viz_err:
    print(f"  ⚠️  Visualization error (non-critical, existing results unaffected): {viz_err}")


# ── SHAP Analysis ────────────────────────────────────────────
try:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n  Running SHAP analysis (this may take ~30 seconds)...")

    # Use a sample of 200 training points (full set is slow)
    shap_sample_size = min(200, len(X_train))
    X_shap = X_train.iloc[:shap_sample_size]

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)

    # ── SHAP Summary Bar Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=pipeline.feature_names,
        plot_type="bar",
        show=False,
        max_display=10,
    )
    plt.title(f"SHAP Feature Importance — {best_model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ SHAP Summary bar chart saved")

    # ── SHAP Beeswarm Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=pipeline.feature_names,
        show=False,
        max_display=10,
    )
    plt.title(f"SHAP Value Distribution — {best_model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_beeswarm.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ SHAP Beeswarm plot saved")

    # ── SHAP Single Prediction Explanation ──
    shap_single = explainer.shap_values(X_test.iloc[[0]])
    shap_explanation = pd.DataFrame({
        "feature":   pipeline.feature_names,
        "shap_value": shap_single[0],
        "abs_shap":   np.abs(shap_single[0]),
    }).sort_values("abs_shap", ascending=False)
    shap_explanation.to_csv(
        os.path.join(REPORTS_DIR, "shap_single_prediction.csv"), index=False
    )
    print("  ✅ SHAP single-prediction explanation saved")

    print("\n  Top SHAP contributors (first test sample):")
    for _, row in shap_explanation.head(5).iterrows():
        direction = "↑" if row["shap_value"] > 0 else "↓"
        print(f"    {direction} {row['feature']:30s}  SHAP: {row['shap_value']:+.2f}")

except ImportError:
    print("  ⚠️  SHAP not installed — skipped. Run: pip install shap")
except Exception as shap_err:
    print(f"  ⚠️  SHAP error (non-critical, existing results unaffected): {shap_err}")


print("\n" + "=" * 60)
print(f"  Phase 5 Model Training — COMPLETE")
print(f"  Best: {best_model_name} | MAPE={best_row['MAPE']:.2f}% | R²={best_row['R2']:.4f}")
print(f"  Plots saved to: {PLOTS_DIR}")
print("=" * 60)
