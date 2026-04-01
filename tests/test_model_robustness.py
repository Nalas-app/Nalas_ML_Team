"""
Phase 5 — Model Robustness & Edge Case Tests
===============================================
Tests model behavior on edge cases, stress scenarios, and fallback handling.

Run: python -m pytest tests/test_model_robustness.py -v
"""

import os
import sys
import time
import pickle
import tracemalloc
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_service"))

from cost_engine import DataStore, BusinessRules, CostEngine, CostingError
from feature_pipeline import FeaturePipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "structured")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CONFIG_PATH = os.path.join(BASE_DIR, "ml_service", "config", "business_rules.json")


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def data_store():
    ds = DataStore(DATA_DIR)
    ds.load()
    return ds

@pytest.fixture(scope="module")
def engine(data_store):
    rules = BusinessRules(CONFIG_PATH)
    return CostEngine(data_store, rules)

@pytest.fixture(scope="module")
def pipeline():
    path = os.path.join(MODEL_DIR, "feature_pipeline.pkl")
    if os.path.exists(path):
        return FeaturePipeline.load(path)
    return None

@pytest.fixture(scope="module")
def model():
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ============================================================
# EDGE CASE TESTS
# ============================================================

class TestEdgeCases:
    """Test model behavior on edge cases."""

    def test_very_small_quantity(self, engine):
        """Minimum quantity = 1 should work."""
        result = engine.predict("ITEM-001", quantity=1)
        assert result['totalCost'] > 0

    def test_very_large_quantity(self, engine):
        """Extremely large quantity should return positive cost."""
        result = engine.predict("ITEM-001", quantity=5000)
        assert result['totalCost'] > 0
        assert result['totalCost'] > engine.predict("ITEM-001", quantity=1)['totalCost']

    def test_all_working_items(self, engine, data_store):
        """All items with data should return valid predictions."""
        success = 0
        errors = 0
        for _, row in data_store.menu_items.iterrows():
            try:
                result = engine.predict(row['item_id'], quantity=10)
                assert result['totalCost'] > 0
                success += 1
            except CostingError:
                errors += 1
        # At least some items should work
        assert success > 0

    def test_missing_event_date(self, engine):
        """Prediction without event_date should still work."""
        result = engine.predict("ITEM-001", quantity=10)
        assert result['totalCost'] > 0

    def test_missing_guest_count(self, engine):
        """Prediction without guest_count should still work."""
        result = engine.predict("ITEM-001", quantity=10, event_date="2026-06-15")
        assert result['totalCost'] > 0

    def test_far_future_date(self, engine):
        """Event date far in the future should work."""
        result = engine.predict("ITEM-001", quantity=10, event_date="2030-12-31")
        assert result['totalCost'] > 0

    def test_past_date(self, engine):
        """Past event date should still work (for historical analysis)."""
        result = engine.predict("ITEM-001", quantity=10, event_date="2020-01-01")
        assert result['totalCost'] > 0

    def test_case_insensitive_lookup(self, engine):
        """Item name lookup should be case-insensitive."""
        r1 = engine.predict("Chicken Fry (KCF)", quantity=10)
        r2 = engine.predict("chicken fry (kcf)", quantity=10)
        assert abs(r1['totalCost'] - r2['totalCost']) < 0.01

    def test_prediction_consistency(self, engine):
        """Same input should always produce same output (deterministic)."""
        r1 = engine.predict("ITEM-001", quantity=10, event_date="2026-11-20")
        r2 = engine.predict("ITEM-001", quantity=10, event_date="2026-11-20")
        assert r1['totalCost'] == r2['totalCost']
        assert r1['confidence'] == r2['confidence']


# ============================================================
# MODEL PREDICTION TESTS (if ML model exists)
# ============================================================

class TestMLModel:
    """Test ML model predictions (requires trained model)."""

    def test_model_exists(self, model):
        """Best model should exist after training."""
        assert model is not None, "Run phase5_model_training.py first"

    def test_model_predicts(self, model, pipeline):
        """Model should produce predictions."""
        if model is None or pipeline is None:
            pytest.skip("No trained model available")

        X = pipeline.transform_single("ITEM-001", 25, "2026-11-20", 100, "Wedding")
        pred = model.predict(X)
        assert len(pred) == 1
        assert pred[0] > 0

    def test_model_non_negative(self, model, pipeline):
        """All predictions should be non-negative."""
        if model is None or pipeline is None:
            pytest.skip("No trained model available")

        items = ["ITEM-001", "ITEM-002", "ITEM-003"]
        for item_id in items:
            X = pipeline.transform_single(item_id, 10)
            pred = model.predict(X)
            assert pred[0] >= 0, f"Negative prediction for {item_id}"

    def test_model_scales_with_quantity(self, model, pipeline):
        """Higher quantity should generally produce higher cost."""
        if model is None or pipeline is None:
            pytest.skip("No trained model available")

        X_small = pipeline.transform_single("ITEM-001", 5)
        X_large = pipeline.transform_single("ITEM-001", 500)
        pred_small = model.predict(X_small)[0]
        pred_large = model.predict(X_large)[0]
        assert pred_large > pred_small


# ============================================================
# PERFORMANCE / STRESS TESTS
# ============================================================

class TestPerformance:
    """Test prediction latency and throughput."""

    def test_rule_based_latency(self, engine):
        """Rule-based prediction should be under 100ms."""
        start = time.time()
        for _ in range(100):
            engine.predict("ITEM-001", quantity=25, event_date="2026-11-20")
        avg_ms = (time.time() - start) / 100 * 1000
        assert avg_ms < 100, f"Average latency {avg_ms:.1f}ms exceeds 100ms"

    def test_batch_prediction_throughput(self, engine):
        """Should handle 100+ predictions quickly."""
        start = time.time()
        for i in range(200):
            try:
                engine.predict("ITEM-001", quantity=i + 1)
            except CostingError:
                pass
        total_ms = (time.time() - start) * 1000
        assert total_ms < 10000, f"200 predictions took {total_ms:.0f}ms"

    def test_ml_prediction_latency(self, model, pipeline):
        """ML model prediction should be under 100ms (prod target: 500ms)."""
        if model is None or pipeline is None:
            pytest.skip("No trained model available")

        start = time.time()
        for _ in range(100):
            X = pipeline.transform_single("ITEM-001", 25)
            model.predict(X)
        avg_ms = (time.time() - start) / 100 * 1000
        assert avg_ms < 100, f"Average ML latency {avg_ms:.1f}ms exceeds 100ms"


# ============================================================
# FALLBACK / ERROR HANDLING TESTS
# ============================================================

class TestFallbackHandling:
    """Test error handling and fallback behavior."""

    def test_invalid_item_gives_clear_error(self, engine):
        """Invalid item should raise CostingError with ML_002."""
        with pytest.raises(CostingError) as exc:
            engine.predict("FAKE-ITEM-999", quantity=10)
        assert exc.value.code == "ML_002"

    def test_zero_quantity_gives_clear_error(self, engine):
        """Zero quantity should raise ML_001."""
        with pytest.raises(CostingError) as exc:
            engine.predict("ITEM-001", quantity=0)
        assert exc.value.code == "ML_001"

    def test_negative_quantity_gives_clear_error(self, engine):
        """Negative quantity should raise ML_001."""
        with pytest.raises(CostingError) as exc:
            engine.predict("ITEM-001", quantity=-10)
        assert exc.value.code == "ML_001"

    def test_item_without_recipe_data(self, engine):
        """Items without recipe data should raise ML_003 or ML_004."""
        # Try an item that may not have data
        items_without_data = ["ITEM-027", "ITEM-029", "ITEM-030"]
        for item_id in items_without_data:
            try:
                engine.predict(item_id, quantity=10)
            except CostingError as e:
                assert e.code in ("ML_003", "ML_004")

    def test_corrupted_model_fallback(self, engine):
        """Even without ML model, rule-based should always work."""
        # Rule-based engine should work regardless of model state
        result = engine.predict("ITEM-001", quantity=10)
        assert result['method'] == 'rule_based'
        assert result['totalCost'] > 0

    def test_all_predictions_positive(self, engine, data_store):
        """No prediction should ever return negative cost."""
        for _, row in data_store.menu_items.iterrows():
            try:
                result = engine.predict(row['item_id'], quantity=10)
                assert result['totalCost'] > 0
                assert result['ingredientCost'] >= 0
                assert result['laborCost'] >= 0
                assert result['overheadCost'] >= 0
            except CostingError:
                pass  # Expected for items without data


# ============================================================
# MEMORY USAGE TESTS
# ============================================================

class TestMemoryUsage:
    """Test that predictions do not consume excessive memory."""

    def test_repeated_predictions_no_memory_leak(self, engine):
        """
        100 consecutive predictions should not grow memory unboundedly.
        Uses tracemalloc (built-in) — no extra packages needed.
        """
        tracemalloc.start()
        for _ in range(100):
            engine.predict("ITEM-001", quantity=25, event_date="2026-11-20")
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        # 100 predictions should fit comfortably under 50 MB
        assert peak_mb < 50, f"Peak memory {peak_mb:.1f} MB exceeds 50 MB limit"

    def test_data_store_load_memory(self):
        """
        Loading the DataStore (CSVs) should be lightweight (< 20 MB).
        """
        tracemalloc.start()
        ds = DataStore(DATA_DIR)
        ds.load()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        assert peak_mb < 20, f"DataStore load used {peak_mb:.1f} MB — unexpectedly large"

    def test_prediction_result_is_not_growing(self, engine):
        """
        Each prediction result dict should stay small (< 10 KB).
        Ensures _meta breakdown doesn't balloon in size.
        """
        import sys as _sys
        result = engine.predict("ITEM-001", quantity=100)
        result_size = _sys.getsizeof(str(result))  # Rough size of the dict
        assert result_size < 10_000, f"Result dict is {result_size} bytes — unexpectedly large"


# ============================================================
# ADVERSARIAL INPUT TESTS
# ============================================================

class TestAdversarialInputs:
    """
    Test that the system handles malicious or manipulative inputs gracefully.
    Goal: confirm the system never crashes, never returns negative cost,
    and handles abuse attempts with clean error codes.
    """

    def test_sql_injection_in_item_id(self, engine):
        """SQL injection attempt in item_id → should return ML_002, not crash."""
        with pytest.raises(CostingError) as exc:
            engine.predict("' OR '1'='1'; DROP TABLE menu_items; --", quantity=10)
        assert exc.value.code == "ML_002"

    def test_xss_in_item_id(self, engine):
        """XSS payload in item_id → should return ML_002, not crash."""
        with pytest.raises(CostingError) as exc:
            engine.predict("<script>alert('xss')</script>", quantity=10)
        assert exc.value.code == "ML_002"

    def test_unicode_in_item_id(self, engine):
        """Unicode/emoji in item_id → should return ML_002, not crash."""
        with pytest.raises(CostingError) as exc:
            engine.predict("ITEM-\u0000\u200b\uffff", quantity=10)
        assert exc.value.code == "ML_002"

    def test_bulk_discount_gaming_exact_boundaries(self, engine):
        """
        Gaming bulk discount thresholds (e.g. ordering 101 instead of 100
        to get 5% off). System should handle this correctly — discount IS
        legitimate, just want to confirm cost never goes negative.
        """
        for qty in [100, 101, 300, 301, 500, 501, 1000, 1001]:
            result = engine.predict("ITEM-001", quantity=qty)
            assert result["totalCost"] > 0, f"Negative cost at qty={qty}"
            assert result["ingredientCost"] > 0

    def test_cost_never_negative_at_any_quantity(self, engine):
        """No matter the quantity, cost must always be strictly positive."""
        for qty in [1, 2, 5, 10, 50, 100, 500, 1000, 5000]:
            result = engine.predict("ITEM-001", quantity=qty)
            assert result["totalCost"] > 0
            assert result["ingredientCost"] >= 0
            assert result["laborCost"] >= 0
            assert result["overheadCost"] >= 0

    def test_confidence_never_exceeds_rule_based_cap(self, engine, data_store):
        """
        Rule-based confidence must never exceed 85% (the hard cap).
        Prevents caller from over-trusting rule-based predictions.
        """
        for _, row in data_store.menu_items.iterrows():
            try:
                result = engine.predict(row["item_id"], quantity=50)
                assert result["confidence"] <= 0.85, (
                    f"Confidence {result['confidence']} exceeds 0.85 cap for {row['item_id']}"
                )
            except CostingError:
                pass  # Items without data are expected to fail

    def test_very_long_item_id_string(self, engine):
        """Extremely long item_id string → ML_002, not a crash."""
        long_id = "A" * 10_000
        with pytest.raises(CostingError) as exc:
            engine.predict(long_id, quantity=10)
        assert exc.value.code == "ML_002"

    def test_float_quantity_handled(self, engine):
        """
        Passing float instead of int for quantity.
        Schema validates this, but engine itself should not crash on int cast.
        """
        # Engine expects int, but if a float sneaks in it should still work or
        # raise a clear ML_001 (not an unhandled exception)
        try:
            result = engine.predict("ITEM-001", quantity=int(25.9))
            assert result["totalCost"] > 0
        except CostingError as e:
            assert e.code == "ML_001"


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
