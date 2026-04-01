"""
Phase 3 — Comprehensive Test Suite for Rule-Based Costing Engine
=================================================================
Tests: unit, integration, validation, edge cases
Run: python -m pytest tests/test_cost_engine.py -v
"""

import os
import sys
import json
import pytest
import time

# Add ml_service to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_service"))

from cost_engine import CostEngine, DataStore, BusinessRules, CostingError
from prediction_logger import PredictionLogger


# ============================================================
# FIXTURES
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "structured")
CONFIG_PATH = os.path.join(BASE_DIR, "ml_service", "config", "business_rules.json")


@pytest.fixture(scope="module")
def data_store():
    """Load data store once for all tests."""
    ds = DataStore(DATA_DIR)
    ds.load()
    return ds


@pytest.fixture(scope="module")
def business_rules():
    """Load business rules once for all tests."""
    return BusinessRules(CONFIG_PATH)


@pytest.fixture(scope="module")
def engine(data_store, business_rules):
    """Create cost engine with real data."""
    return CostEngine(data_store, business_rules)


@pytest.fixture
def pred_logger(tmp_path):
    """Create prediction logger with temp directory."""
    return PredictionLogger(str(tmp_path))


# ============================================================
# UNIT TESTS — Data Store
# ============================================================

class TestDataStore:
    """Test data loading and lookup functions."""

    def test_data_loads_successfully(self, data_store):
        """Data store should load without errors."""
        assert data_store.is_loaded is True

    def test_menu_items_loaded(self, data_store):
        """Should have 34 menu items."""
        assert len(data_store.menu_items) == 34

    def test_ingredients_loaded(self, data_store):
        """Should have 129 ingredients."""
        assert len(data_store.ingredients) == 129

    def test_recipes_loaded(self, data_store):
        """Should have recipe rows."""
        assert len(data_store.recipes) > 0

    def test_get_menu_item_by_id(self, data_store):
        """Should find item by ID."""
        item = data_store.get_menu_item("ITEM-001")
        assert item is not None
        assert item['item_name'] == "Chicken Fry (KCF)"

    def test_get_menu_item_by_name(self, data_store):
        """Should find item by name (case-insensitive)."""
        item = data_store.get_menu_item("chicken fry (kcf)")
        assert item is not None
        assert item['item_id'] == "ITEM-001"

    def test_get_menu_item_not_found(self, data_store):
        """Should return None for unknown item."""
        item = data_store.get_menu_item("NONEXISTENT-ITEM")
        assert item is None

    def test_get_recipe(self, data_store):
        """Should return recipe rows for a valid item."""
        recipe = data_store.get_recipe("ITEM-001")
        assert len(recipe) > 0

    def test_get_recipe_empty(self, data_store):
        """Should return empty df for unknown item."""
        recipe = data_store.get_recipe("UNKNOWN-ID")
        assert len(recipe) == 0

    def test_get_ingredient_price(self, data_store):
        """Should return a positive price for valid ingredient."""
        price = data_store.get_ingredient_price("ING-002")  # Basmati Rice
        assert price is not None
        assert price > 0

    def test_get_ingredient_price_unmatched(self, data_store):
        """Should return None for UNMATCHED."""
        price = data_store.get_ingredient_price("UNMATCHED")
        assert price is None


# ============================================================
# UNIT TESTS — Business Rules
# ============================================================

class TestBusinessRules:
    """Test business rule calculations."""

    def test_labor_percentage(self, business_rules):
        assert business_rules.labor_pct == 0.15

    def test_overhead_percentage(self, business_rules):
        assert business_rules.overhead_pct == 0.10

    def test_wastage_percentage(self, business_rules):
        assert business_rules.wastage_pct == 0.05

    def test_profit_percentage(self, business_rules):
        assert business_rules.profit_pct == 0.10

    def test_seasonal_adjustment_wedding_season(self, business_rules):
        """October is wedding season — should have 15% adjustment."""
        adj = business_rules.get_seasonal_adjustment("2026-10-15")
        assert adj == 0.15

    def test_seasonal_adjustment_off_season(self, business_rules):
        """July is off-season — no adjustment."""
        adj = business_rules.get_seasonal_adjustment("2026-07-15")
        assert adj == 0.0

    def test_seasonal_adjustment_no_date(self, business_rules):
        """No date provided — no adjustment."""
        adj = business_rules.get_seasonal_adjustment(None)
        assert adj == 0.0

    def test_bulk_discount_small(self, business_rules):
        """< 100 qty should have no discount."""
        discount = business_rules.get_bulk_discount(50)
        assert discount == 0.0

    def test_bulk_discount_medium(self, business_rules):
        """101-300 qty should have 5% discount."""
        discount = business_rules.get_bulk_discount(200)
        assert discount == 0.05

    def test_bulk_discount_large(self, business_rules):
        """501-1000 qty should have 10% discount."""
        discount = business_rules.get_bulk_discount(700)
        assert discount == 0.10

    def test_min_cost_nonveg(self, business_rules):
        min_cost = business_rules.get_min_cost("Non-Veg Main Course")
        assert min_cost == 100

    def test_min_cost_veg(self, business_rules):
        min_cost = business_rules.get_min_cost("Veg Main Course")
        assert min_cost == 50

    def test_min_cost_unknown_category(self, business_rules):
        min_cost = business_rules.get_min_cost("Unknown Category")
        assert min_cost == 40  # default


# ============================================================
# UNIT TESTS — Cost Engine Predictions
# ============================================================

class TestCostEngine:
    """Test the core prediction logic."""

    def test_predict_valid_item(self, engine):
        """Should return valid cost breakdown for KCF."""
        result = engine.predict("ITEM-001", quantity=1)

        assert result["ingredientCost"] > 0
        assert result["laborCost"] > 0
        assert result["overheadCost"] > 0
        assert result["totalCost"] > 0
        assert 0 < result["confidence"] <= 1
        assert result["method"] == "rule_based"
        assert result["modelVersion"] == "v0.1.0"

    def test_predict_by_name(self, engine):
        """Should work with item name instead of ID."""
        result = engine.predict("Chicken Fry (KCF)", quantity=1)
        assert result["totalCost"] > 0

    def test_all_costs_positive(self, engine):
        """All cost components must be positive (business constraint)."""
        result = engine.predict("ITEM-001", quantity=1)

        assert result["ingredientCost"] > 0
        assert result["laborCost"] > 0
        assert result["overheadCost"] > 0
        assert result["totalCost"] > 0

    def test_labor_is_15_percent(self, engine):
        """Labor cost must be exactly 15% of ingredient cost."""
        result = engine.predict("ITEM-001", quantity=1)

        expected_labor = result["ingredientCost"] * 0.15
        assert abs(result["laborCost"] - expected_labor) < 0.01

    def test_overhead_is_10_percent(self, engine):
        """Overhead cost must be exactly 10% of ingredient cost."""
        result = engine.predict("ITEM-001", quantity=1)

        expected_overhead = result["ingredientCost"] * 0.10
        assert abs(result["overheadCost"] - expected_overhead) < 0.01

    def test_cost_scales_with_quantity(self, engine):
        """Double quantity should approximately double cost."""
        result_1 = engine.predict("ITEM-001", quantity=1)
        result_2 = engine.predict("ITEM-001", quantity=2)

        # Should be close to 2x (may not be exact due to min costs / discounts)
        ratio = result_2["totalCost"] / result_1["totalCost"]
        assert 1.5 < ratio < 2.5  # Allow some variance

    def test_cost_breakdown_sums_correctly(self, engine):
        """Total cost should equal sum of components + profit."""
        result = engine.predict("ITEM-001", quantity=1)
        meta = result["_meta"]

        ing = result["ingredientCost"]
        labor = result["laborCost"]
        overhead = result["overheadCost"]
        wastage = ing * 0.05
        subtotal = ing + labor + overhead + wastage

        seasonal = meta["seasonal_adjustment"]
        if seasonal > 0:
            subtotal *= (1 + seasonal)

        profit = subtotal * 0.10
        expected_total = subtotal + profit

        bulk = meta["bulk_discount"]
        if bulk > 0:
            expected_total *= (1 - bulk)

        assert abs(result["totalCost"] - expected_total) < 1.0  # Within Rs.1

    def test_confidence_between_0_and_1(self, engine):
        """Confidence should always be between 0 and 1."""
        result = engine.predict("ITEM-001", quantity=1)
        assert 0 < result["confidence"] <= 1

    def test_method_is_rule_based(self, engine):
        """Method should be 'rule_based' in Phase 3."""
        result = engine.predict("ITEM-001", quantity=1)
        assert result["method"] == "rule_based"


# ============================================================
# ERROR HANDLING TESTS
# ============================================================

class TestErrorHandling:
    """Test that errors are raised correctly."""

    def test_zero_quantity_raises_error(self, engine):
        """Zero quantity should raise ML_001."""
        with pytest.raises(CostingError) as exc_info:
            engine.predict("ITEM-001", quantity=0)
        assert exc_info.value.code == "ML_001"

    def test_negative_quantity_raises_error(self, engine):
        """Negative quantity should raise ML_001."""
        with pytest.raises(CostingError) as exc_info:
            engine.predict("ITEM-001", quantity=-5)
        assert exc_info.value.code == "ML_001"

    def test_invalid_item_raises_error(self, engine):
        """Invalid menu item should raise ML_002."""
        with pytest.raises(CostingError) as exc_info:
            engine.predict("NONEXISTENT-ITEM-XYZ", quantity=10)
        assert exc_info.value.code == "ML_002"

    def test_item_without_data_raises_error(self, engine):
        """Items without calculable costs should raise ML_003 or ML_004."""
        # ITEM-008 = Rasam (Menu file, no quantities)
        with pytest.raises(CostingError) as exc_info:
            engine.predict("ITEM-008", quantity=1)
        assert exc_info.value.code in ("ML_003", "ML_004")

    def test_error_has_code_and_message(self, engine):
        """CostingError should have code, message, and details."""
        with pytest.raises(CostingError) as exc_info:
            engine.predict("FAKE-ITEM", quantity=1)

        error = exc_info.value
        assert hasattr(error, 'code')
        assert hasattr(error, 'message')
        assert hasattr(error, 'details')
        assert len(error.message) > 0


# ============================================================
# EDGE CASE TESTS
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_quantity_one(self, engine):
        """Single serving should work."""
        result = engine.predict("ITEM-001", quantity=1)
        assert result["totalCost"] > 0

    def test_very_large_quantity(self, engine):
        """Very large quantity should apply bulk discount."""
        result = engine.predict("ITEM-001", quantity=1000)
        assert result["totalCost"] > 0
        assert "bulk_discount" in str(result["_meta"]["flags"])

    def test_wedding_season_costs_more(self, engine):
        """Prediction during wedding season should cost more."""
        result_normal = engine.predict("ITEM-001", quantity=10, event_date="2026-07-15")
        result_wedding = engine.predict("ITEM-001", quantity=10, event_date="2026-11-15")

        assert result_wedding["totalCost"] > result_normal["totalCost"]

    def test_ghee_rice_prediction(self, engine):
        """Ghee Rice (ITEM-003) should return reasonable cost."""
        result = engine.predict("ITEM-003", quantity=1)
        # Ghee Rice for 1 batch should be in hundreds, not lakhs
        assert result["totalCost"] < 10000

    def test_chicken_curry_prediction(self, engine):
        """Chicken Curry should have valid cost."""
        result = engine.predict("ITEM-002", quantity=1)
        assert result["totalCost"] > 0

    def test_prediction_latency(self, engine):
        """Prediction should be fast (< 100ms for rule-based)."""
        start = time.time()
        engine.predict("ITEM-001", quantity=10)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 100  # Should be well under 100ms


# ============================================================
# PREDICTION LOGGER TESTS
# ============================================================

class TestPredictionLogger:
    """Test prediction logging."""

    def test_log_successful_prediction(self, pred_logger):
        """Should log successful prediction."""
        pred_logger.log_prediction(
            request={"menuItemId": "ITEM-001", "quantity": 10},
            response={"totalCost": 1000, "confidence": 0.85, "method": "rule_based"},
            latency_ms=15.5,
        )
        metrics = pred_logger.get_metrics()
        assert metrics["total_predictions"] == 1

    def test_log_error_prediction(self, pred_logger):
        """Should log error prediction."""
        pred_logger.log_prediction(
            request={"menuItemId": "FAKE", "quantity": 10},
            response={},
            latency_ms=2.0,
            error={"code": "ML_002", "message": "Item not found"},
        )
        metrics = pred_logger.get_metrics()
        assert metrics["error_count"] == 1

    def test_metrics_accurate(self, pred_logger):
        """Metrics should reflect actual predictions."""
        for i in range(5):
            pred_logger.log_prediction(
                request={"menuItemId": "ITEM-001", "quantity": i + 1},
                response={"totalCost": 100 * (i + 1)},
                latency_ms=10.0 + i,
            )
        metrics = pred_logger.get_metrics()
        assert metrics["total_predictions"] == 5
        assert metrics["avg_latency_ms"] > 0

    def test_log_file_created(self, pred_logger, tmp_path):
        """JSONL log file should be created."""
        pred_logger.log_prediction(
            request={"menuItemId": "ITEM-001", "quantity": 1},
            response={"totalCost": 500},
            latency_ms=5.0,
        )
        log_files = list(tmp_path.glob("predictions_*.jsonl"))
        assert len(log_files) > 0


# ============================================================
# INTEGRATION TEST — Full Flow
# ============================================================

class TestFullFlow:
    """Test the complete prediction flow end-to-end."""

    def test_end_to_end_prediction(self, engine):
        """Full prediction flow should work."""
        # Step 1: Predict
        result = engine.predict(
            menu_item_id="ITEM-001",
            quantity=25,
            event_date="2026-11-20",
            guest_count=100,
        )

        # Step 2: Verify response structure
        assert "ingredientCost" in result
        assert "laborCost" in result
        assert "overheadCost" in result
        assert "totalCost" in result
        assert "confidence" in result
        assert "modelVersion" in result
        assert "method" in result
        assert "_meta" in result

        # Step 3: Verify business rules (allow small rounding tolerance)
        assert abs(result["laborCost"] - round(result["ingredientCost"] * 0.15, 2)) < 0.02
        assert abs(result["overheadCost"] - round(result["ingredientCost"] * 0.10, 2)) < 0.02

        # Step 4: Verify all costs positive
        assert result["ingredientCost"] > 0
        assert result["laborCost"] > 0
        assert result["overheadCost"] > 0
        assert result["totalCost"] > 0

        # Step 5: Verify seasonal adjustment applied (November = wedding)
        assert "seasonal_adjustment" in str(result["_meta"]["flags"])

    def test_multiple_items_no_crash(self, engine):
        """Predicting multiple items shouldn't crash."""
        working_items = ["ITEM-001", "ITEM-002", "ITEM-003"]
        for item_id in working_items:
            result = engine.predict(item_id, quantity=5)
            assert result["totalCost"] > 0


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
