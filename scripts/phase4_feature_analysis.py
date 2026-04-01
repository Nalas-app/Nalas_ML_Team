"""
Phase 4 — Feature Validation & Selection Script
=================================================
Runs the full feature pipeline on available data, generates:
    - Feature correlation matrix
    - VIF analysis
    - Feature importance (using synthetic training)
    - Feature selection recommendations
    - Performance benchmarks

Run: python scripts/phase4_feature_analysis.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "ml_service"))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from cost_engine import DataStore, BusinessRules
from feature_pipeline import FeaturePipeline, FeatureExtractor, FeatureAnalyzer

# ============================================================
# PATHS
# ============================================================
DATA_DIR = os.path.join(BASE_DIR, "data", "structured")
CONFIG_PATH = os.path.join(BASE_DIR, "ml_service", "config", "business_rules.json")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "feature_analysis")
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("  NALAS ML COSTING — Phase 4 Feature Analysis")
print("=" * 60)


# ============================================================
# STEP 1: Load Data
# ============================================================
print("\n[1/6] Loading data...")
data_store = DataStore(DATA_DIR)
data_store.load()

print(f"  ✅ {len(data_store.menu_items)} menu items")
print(f"  ✅ {len(data_store.ingredients)} ingredients")
print(f"  ✅ {len(data_store.recipes)} recipe rows")


# ============================================================
# STEP 2: Generate synthetic training data
# ============================================================
print("\n[2/6] Generating synthetic training data...")

# Create a synthetic dataset using all available menu items
np.random.seed(42)
n_samples = 500

# Get items that have calculable costs
working_items = []
for _, row in data_store.menu_items.iterrows():
    item_id = row['item_id']
    recipe = data_store.get_recipe(item_id)
    has_qty = recipe['quantity_per_base_unit'].notna().sum()
    if has_qty > 0:
        working_items.append(item_id)

print(f"  Items with calculable costs: {len(working_items)}")

# Generate synthetic orders
event_types = ['Wedding', 'Birthday', 'Corporate', 'Other']
months = list(range(1, 13))

records = []
for i in range(n_samples):
    item_id = np.random.choice(working_items)
    qty = np.random.randint(5, 500)
    month = np.random.choice(months)
    day = np.random.randint(1, 28)
    year = 2026
    event_date = f"{year}-{month:02d}-{day:02d}"
    guest_count = np.random.randint(20, 500)
    event_type = np.random.choice(event_types)

    records.append({
        'menu_item_id': item_id,
        'quantity': qty,
        'event_date': event_date,
        'guest_count': guest_count,
        'event_type': event_type,
    })

train_data = pd.DataFrame(records)
print(f"  ✅ Generated {len(train_data)} synthetic training samples")


# ============================================================
# STEP 3: Extract features & benchmark performance
# ============================================================
print("\n[3/6] Extracting features...")

pipeline = FeaturePipeline(data_store=data_store)

# Benchmark extraction time
start = time.time()
raw_features = pipeline.extract_features(train_data)
extraction_time = (time.time() - start) * 1000

# Benchmark fit time
start = time.time()
pipeline.fit(raw_features)
fit_time = (time.time() - start) * 1000

# Benchmark transform time
start = time.time()
X = pipeline.transform(raw_features)
transform_time = (time.time() - start) * 1000

# Benchmark single prediction
start = time.time()
for _ in range(100):
    pipeline.transform_single("ITEM-001", 25, "2026-11-20", 100, "Wedding")
single_pred_time = (time.time() - start) * 1000 / 100

print(f"  Raw features shape: {raw_features.shape}")
print(f"  Transformed features shape: {X.shape}")
print(f"  Feature names: {pipeline.feature_names}")

performance = {
    "extraction_time_ms": round(extraction_time, 2),
    "fit_time_ms": round(fit_time, 2),
    "transform_time_ms": round(transform_time, 2),
    "single_prediction_ms": round(single_pred_time, 2),
    "samples": n_samples,
    "raw_features": raw_features.shape[1],
    "transformed_features": X.shape[1],
}
print(f"\n  Performance:")
print(f"    Extraction (500 samples): {performance['extraction_time_ms']:.1f}ms")
print(f"    Fit: {performance['fit_time_ms']:.1f}ms")
print(f"    Transform: {performance['transform_time_ms']:.1f}ms")
print(f"    Single prediction: {performance['single_prediction_ms']:.2f}ms")


# ============================================================
# STEP 4: Calculate target variable (for analysis)
# ============================================================
print("\n[4/6] Calculating target variable for analysis...")

from cost_engine import CostEngine
rules = BusinessRules(CONFIG_PATH)
engine = CostEngine(data_store, rules)

targets = []
for _, row in train_data.iterrows():
    try:
        result = engine.predict(
            row['menu_item_id'],
            int(row['quantity']),
            row.get('event_date'),
            row.get('guest_count')
        )
        targets.append(result['totalCost'])
    except Exception:
        targets.append(np.nan)

y = pd.Series(targets, name='total_cost')
valid_mask = y.notna()
X_valid = X[valid_mask].reset_index(drop=True)
y_valid = y[valid_mask].reset_index(drop=True)

print(f"  ✅ {len(y_valid)} samples with valid costs")
print(f"  Cost range: Rs.{y_valid.min():.2f} — Rs.{y_valid.max():.2f}")
print(f"  Cost mean: Rs.{y_valid.mean():.2f}")


# ============================================================
# STEP 5: Feature Analysis
# ============================================================
print("\n[5/6] Running feature analysis...")

analyzer = FeatureAnalyzer()

# 5a. Correlation matrix
print("  Computing correlation matrix...")
corr_matrix = analyzer.calculate_correlation_matrix(X_valid)
corr_matrix.to_csv(os.path.join(REPORTS_DIR, "correlation_matrix.csv"))

# 5b. High correlations
high_corr = analyzer.find_high_correlations(corr_matrix, threshold=0.9)
print(f"  High correlations (>0.9): {len(high_corr)}")
for hc in high_corr:
    print(f"    {hc['feature_1']} <-> {hc['feature_2']}: {hc['correlation']}")

# 5c. VIF analysis
print("  Computing VIF...")
vif_df = analyzer.calculate_vif(X_valid)
vif_df.to_csv(os.path.join(REPORTS_DIR, "vif_analysis.csv"), index=False)
print("  VIF results:")
for _, row in vif_df.iterrows():
    status = "⚠️ HIGH" if row['VIF'] > 10 else "✅ OK"
    print(f"    {row['feature']:30s} VIF={row['VIF']:8.2f}  {status}")

# 5d. Train a quick model for feature importance
print("\n  Training RandomForest for feature importance...")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_valid, y_valid)

# Cross-validation score
cv_scores = cross_val_score(rf, X_valid, y_valid, cv=5, scoring='r2')
print(f"  R² (5-fold CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

mape_scores = cross_val_score(
    rf, X_valid, y_valid, cv=5,
    scoring='neg_mean_absolute_percentage_error'
)
print(f"  MAPE (5-fold CV): {-mape_scores.mean()*100:.2f}% ± {mape_scores.std()*100:.2f}%")

# 5e. Feature importance
importance_df = analyzer.extract_feature_importance(rf, pipeline.feature_names)
importance_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False)

print("\n  Feature Importance (Top 15):")
print(f"  {'Rank':>4} {'Feature':30s} {'Importance':>12} {'Cumulative':>12}")
print("  " + "-" * 62)
for _, row in importance_df.head(15).iterrows():
    print(f"  {int(row['rank']):4d} {row['feature']:30s} {row['importance_pct']:10.2f}% {row['cumulative_pct']:10.2f}%")

# 5f. Validate top features
validations = analyzer.validate_top_features(importance_df)
print("\n  Feature Validation:")
for v in validations:
    icon = "✅" if v['status'] == 'PASS' else ("⚠️" if v['status'] == 'WARNING' else "❌")
    print(f"    {icon} {v['check']}: {v['details']}")


# ============================================================
# STEP 6: Feature Selection Recommendation
# ============================================================
print("\n[6/6] Feature selection recommendation...")

# Selected features: top features by importance + business-critical features
selected = importance_df[importance_df['cumulative_pct'] <= 95].head(15)
selected_names = selected['feature'].tolist()

# Always include business-critical features even if low importance
critical_features = ['quantity', 'is_wedding_season']
for feat in critical_features:
    matches = [f for f in pipeline.feature_names if feat in f]
    for m in matches:
        if m not in selected_names:
            selected_names.append(m)

print(f"\n  Selected {len(selected_names)} features (from {len(pipeline.feature_names)} total):")
for i, name in enumerate(selected_names, 1):
    imp = importance_df[importance_df['feature'] == name]['importance_pct']
    pct = f"{imp.values[0]:.2f}%" if len(imp) > 0 else "N/A"
    print(f"    {i:2d}. {name:30s} ({pct})")


# ============================================================
# SAVE REPORTS
# ============================================================
print("\n  Saving reports...")

# Save full analysis report
report = {
    'generated_at': datetime.now().isoformat(),
    'data_summary': {
        'menu_items': len(data_store.menu_items),
        'ingredients': len(data_store.ingredients),
        'working_items': len(working_items),
        'training_samples': n_samples,
        'valid_samples': len(y_valid),
    },
    'performance_benchmarks': performance,
    'feature_counts': {
        'raw_features': int(raw_features.shape[1]),
        'transformed_features': int(X.shape[1]),
        'selected_features': len(selected_names),
    },
    'model_performance': {
        'r2_mean': round(float(cv_scores.mean()), 4),
        'r2_std': round(float(cv_scores.std()), 4),
        'mape_mean': round(float(-mape_scores.mean() * 100), 2),
        'mape_std': round(float(mape_scores.std() * 100), 2),
    },
    'high_correlations': high_corr,
    'selected_features': selected_names,
    'feature_importance_top10': importance_df.head(10).to_dict(orient='records'),
    'validations': validations,
}

with open(os.path.join(REPORTS_DIR, "feature_analysis_report.json"), 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Save the fitted pipeline
pipeline_path = os.path.join(BASE_DIR, "model", "feature_pipeline.pkl")
pipeline.save(pipeline_path)
print(f"  ✅ Pipeline saved to {pipeline_path}")

# Save raw features for reference
raw_features.to_csv(os.path.join(REPORTS_DIR, "raw_features_sample.csv"), index=False)

print(f"\n  Reports saved to: {REPORTS_DIR}")

print("\n" + "=" * 60)
print("  Phase 4 Feature Analysis — COMPLETE")
print("=" * 60)
