"""
Phase 6 — Production Readiness Tests
=======================================
Tests: model registry, production predictor, fallback logic, performance.

Run: python -m pytest tests/test_production.py -v
"""

import os
import sys
import time
import json
import pickle
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_service"))

from cost_engine import DataStore, BusinessRules, CostEngine, CostingError
from feature_pipeline import FeaturePipeline
from model_registry import ModelRegistry
from production_predictor import ProductionPredictor

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
def predictor():
    """Production predictor with real data."""
    p = ProductionPredictor(MODEL_DIR, DATA_DIR, CONFIG_PATH)
    p.initialize()
    return p


@pytest.fixture
def temp_registry(tmp_path):
    """Fresh model registry in temp directory."""
    return ModelRegistry(str(tmp_path / "model"))


@pytest.fixture
def dummy_model():
    """Simple model for testing registry."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(np.array([[1], [2], [3]]), np.array([10, 20, 30]))
    return model


@pytest.fixture
def dummy_pipeline(data_store):
    """Fitted pipeline for testing."""
    pipeline = FeaturePipeline(data_store=data_store)
    data = pd.DataFrame([{
        'menu_item_id': 'ITEM-001', 'quantity': 10,
        'event_date': '2026-06-15', 'guest_count': 50, 'event_type': 'Other'
    }])
    pipeline.fit_transform(data)
    return pipeline


# ============================================================
# MODEL REGISTRY TESTS
# ============================================================

class TestModelRegistry:

    def test_create_registry(self, temp_registry):
        """Registry should initialize cleanly."""
        assert temp_registry.get_active_version() is None
        assert len(temp_registry.list_versions()) == 0

    def test_register_model(self, temp_registry, dummy_model, dummy_pipeline):
        """Should register a model version."""
        temp_registry.register_model(
            model=dummy_model,
            pipeline=dummy_pipeline,
            version="v1.0.0",
            algorithm="LinearRegression",
            training_samples=100,
            metrics={"mape": 10.0, "r2": 0.90},
        )
        assert temp_registry.get_active_version() == "v1.0.0"
        assert len(temp_registry.list_versions()) == 1

    def test_register_multiple_versions(self, temp_registry, dummy_model, dummy_pipeline):
        """Should track multiple versions."""
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v1.0.0",
            metrics={"mape": 12.0}, set_active=True
        )
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v1.1.0",
            metrics={"mape": 8.0}, set_active=True
        )
        assert temp_registry.get_active_version() == "v1.1.0"
        assert len(temp_registry.list_versions()) == 2

    def test_load_model(self, temp_registry, dummy_model, dummy_pipeline):
        """Should load a registered model."""
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v1.0.0"
        )
        loaded = temp_registry.load_model("v1.0.0")
        assert loaded["version"] == "v1.0.0"
        assert loaded["model"] is not None
        assert loaded["metadata"]["modelVersion"] == "v1.0.0"

    def test_load_active_model(self, temp_registry, dummy_model, dummy_pipeline):
        """Should load active model when no version specified."""
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v2.0.0", set_active=True
        )
        loaded = temp_registry.load_model()
        assert loaded["version"] == "v2.0.0"

    def test_rollback(self, temp_registry, dummy_model, dummy_pipeline):
        """Should rollback to previous version."""
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v1.0.0",
            metrics={"mape": 10.0}
        )
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v2.0.0",
            metrics={"mape": 15.0}
        )
        assert temp_registry.get_active_version() == "v2.0.0"

        temp_registry.rollback("v1.0.0")
        assert temp_registry.get_active_version() == "v1.0.0"

    def test_should_deploy_improvement(self, temp_registry, dummy_model, dummy_pipeline):
        """Should recommend deploy if MAPE improves."""
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v1.0.0",
            metrics={"mape": 12.0}
        )
        decision = temp_registry.should_deploy({"mape": 8.0})
        assert decision["deploy"] is True

    def test_should_not_deploy_degradation(self, temp_registry, dummy_model, dummy_pipeline):
        """Should NOT recommend deploy if MAPE degrades significantly."""
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v1.0.0",
            metrics={"mape": 10.0}
        )
        decision = temp_registry.should_deploy({"mape": 15.0})
        assert decision["deploy"] is False

    def test_load_nonexistent_version(self, temp_registry):
        """Should raise FileNotFoundError for unknown version."""
        with pytest.raises(FileNotFoundError):
            temp_registry.load_model("v99.99.99")

    def test_metadata_saved(self, temp_registry, dummy_model, dummy_pipeline):
        """Metadata should include all fields."""
        temp_registry.register_model(
            dummy_model, dummy_pipeline, "v1.0.0",
            algorithm="TestAlgo",
            training_samples=500,
            features=["f1", "f2"],
            metrics={"mape": 5.0},
            limitations=["Test limitation"],
        )
        loaded = temp_registry.load_model("v1.0.0")
        meta = loaded["metadata"]
        assert meta["algorithm"] == "TestAlgo"
        assert meta["trainingDataSize"] == 500
        assert "f1" in meta["features"]
        assert meta["performanceMetrics"]["mape"] == 5.0
        assert "Test limitation" in meta["limitations"]


# ============================================================
# PRODUCTION PREDICTOR TESTS
# ============================================================

class TestProductionPredictor:

    def test_predictor_initializes(self, predictor):
        """Should initialize successfully."""
        assert predictor._initialized is True

    def test_predict_valid_item(self, predictor):
        """Should predict for valid item."""
        result = predictor.predict("ITEM-001", 25)
        assert result["totalCost"] > 0
        assert result["method"] in ("ml_model", "rule_based")

    def test_predict_returns_breakdown(self, predictor):
        """Should return full cost breakdown."""
        result = predictor.predict("ITEM-001", 10)
        assert "ingredientCost" in result
        assert "laborCost" in result
        assert "overheadCost" in result
        assert "totalCost" in result
        assert "confidence" in result
        assert "modelVersion" in result
        assert "method" in result

    def test_predict_by_name(self, predictor):
        """Should work with item name."""
        result = predictor.predict("Chicken Fry (KCF)", 10)
        assert result["totalCost"] > 0

    def test_force_rule_based(self, predictor):
        """Should use rule-based when forced."""
        result = predictor.predict("ITEM-001", 10, force_method="rule_based")
        assert result["method"] == "rule_based"

    def test_invalid_item_raises_error(self, predictor):
        """Invalid item should raise error (from either method)."""
        with pytest.raises(CostingError):
            predictor.predict("FAKE-ITEM-999", 10)

    def test_zero_quantity_raises_error(self, predictor):
        """Zero quantity should raise error."""
        with pytest.raises(CostingError):
            predictor.predict("ITEM-001", 0)

    def test_all_costs_positive(self, predictor):
        """All cost components must be positive."""
        result = predictor.predict("ITEM-001", 25, "2026-11-20", 100, "Wedding")
        assert result["ingredientCost"] > 0
        assert result["laborCost"] > 0
        assert result["overheadCost"] > 0
        assert result["totalCost"] > 0

    def test_confidence_range(self, predictor):
        """Confidence should be between 0.1 and 0.95."""
        result = predictor.predict("ITEM-001", 10)
        assert 0.1 <= result["confidence"] <= 0.95

    def test_seasonal_date(self, predictor):
        """Wedding season should produce higher cost."""
        r_normal = predictor.predict("ITEM-001", 10, "2026-07-15")
        r_wedding = predictor.predict("ITEM-001", 10, "2026-11-15")
        # At least rule-based should show difference
        if r_normal["method"] == "rule_based" and r_wedding["method"] == "rule_based":
            assert r_wedding["totalCost"] >= r_normal["totalCost"]

    def test_stats_tracking(self, predictor):
        """Should track prediction statistics."""
        predictor.predict("ITEM-001", 10)
        stats = predictor.get_stats()
        assert stats["total_predictions"] > 0
        assert "ml_rate_pct" in stats


# ============================================================
# FALLBACK LOGIC TESTS
# ============================================================

class TestFallbackLogic:

    def test_fallback_works_without_ml(self):
        """Predictor should work even without ML model."""
        p = ProductionPredictor(
            os.path.join(tempfile.gettempdir(), "nonexistent_model"),
            DATA_DIR, CONFIG_PATH
        )
        p.initialize()
        assert p.is_ml_available is False

        result = p.predict("ITEM-001", 10)
        assert result["method"] == "rule_based"
        assert result["totalCost"] > 0

    def test_fallback_on_forced_rule_based(self, predictor):
        """Forced rule-based should always work."""
        result = predictor.predict("ITEM-001", 10, force_method="rule_based")
        assert result["method"] == "rule_based"
        assert "_meta" in result
        assert "fallback_reason" in result["_meta"]


# ============================================================
# PRODUCTION PERFORMANCE TESTS
# ============================================================

class TestProductionPerformance:

    def test_prediction_latency(self, predictor):
        """Production prediction < 500ms (99th percentile target)."""
        latencies = []
        for _ in range(50):
            start = time.time()
            predictor.predict("ITEM-001", 25)
            latencies.append((time.time() - start) * 1000)

        p99 = np.percentile(latencies, 99)
        assert p99 < 500, f"P99 latency {p99:.1f}ms exceeds 500ms target"

    def test_concurrent_predictions(self, predictor):
        """Should handle sequential predictions without errors."""
        errors = 0
        for i in range(100):
            try:
                predictor.predict("ITEM-001", i + 1)
            except CostingError:
                pass
            except Exception:
                errors += 1
        assert errors == 0

    def test_mixed_predictions(self, predictor):
        """Should handle mix of valid and invalid predictions."""
        items = ["ITEM-001", "FAKE", "ITEM-002", "ITEM-003", "NONEXIST"]
        for item in items:
            try:
                result = predictor.predict(item, 10)
                assert result["totalCost"] > 0
            except CostingError:
                pass  # Expected for invalid items


# ============================================================
# VALIDATION TESTS
# ============================================================

class TestValidation:

    def test_prediction_reasonable_range(self, predictor):
        """Predictions should be in reasonable range."""
        result = predictor.predict("ITEM-001", 10)
        # 10 servings of KCF should not be millions
        assert result["totalCost"] < 1_000_000

    def test_labor_is_15_percent(self, predictor):
        """Labor should be ~15% of ingredient cost (rule-based)."""
        result = predictor.predict("ITEM-001", 10, force_method="rule_based")
        expected = result["ingredientCost"] * 0.15
        assert abs(result["laborCost"] - expected) < 1.0

    def test_overhead_is_10_percent(self, predictor):
        """Overhead should be ~10% of ingredient cost (rule-based)."""
        result = predictor.predict("ITEM-001", 10, force_method="rule_based")
        expected = result["ingredientCost"] * 0.10
        assert abs(result["overheadCost"] - expected) < 1.0


# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
