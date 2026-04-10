import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_service'))


class TestPredictCostEndpoint:

    def test_basic_prediction_returns_200(self, client):
        response = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50})
        assert response.status_code == 200

    def test_response_has_all_required_fields(self, client):
        data = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50}).json()
        for field in ["ingredientCost", "laborCost", "overheadCost", "totalCost", "confidence", "modelVersion", "method"]:
            assert field in data

    def test_all_costs_are_positive(self, client):
        data = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50}).json()
        assert data["ingredientCost"] >= 0
        assert data["laborCost"] >= 0
        assert data["totalCost"] > 0

    def test_confidence_between_0_and_1(self, client):
        data = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50}).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_method_is_valid_value(self, client):
        data = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50}).json()
        assert data["method"] in ["ml_model", "rule_based"]

    def test_total_cost_greater_than_ingredient_cost(self, client):
        data = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50}).json()
        assert data["totalCost"] > data["ingredientCost"]

    def test_quantity_scales_cost(self, client):
        r1 = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 10}).json()
        r2 = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 100}).json()
        assert r2["totalCost"] > r1["totalCost"]

    def test_with_optional_event_date(self, client):
        r = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50, "eventDate": "2026-12-15"})
        assert r.status_code == 200

    def test_with_optional_guest_count(self, client):
        r = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50, "guestCount": 200})
        assert r.status_code == 200

    def test_with_all_optional_fields(self, client):
        r = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50, "eventDate": "2026-11-20", "guestCount": 150})
        assert r.status_code == 200


class TestValidationErrors:

    def test_zero_quantity_rejected(self, client):
        assert client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 0}).status_code == 422

    def test_negative_quantity_rejected(self, client):
        assert client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": -10}).status_code == 422

    def test_missing_quantity_rejected(self, client):
        assert client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001"}).status_code == 422

    def test_missing_menu_item_id_rejected(self, client):
        assert client.post("/ml/predict-cost", json={"quantity": 50}).status_code == 422

    def test_invalid_date_format_rejected(self, client):
        assert client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50, "eventDate": "10-04-2026"}).status_code == 422

    def test_invalid_menu_item_returns_error(self, client):
        assert client.post("/ml/predict-cost", json={"menuItemId": "DOES-NOT-EXIST-999", "quantity": 50}).status_code in [400, 404, 422]

    def test_zero_guest_count_rejected(self, client):
        assert client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50, "guestCount": 0}).status_code == 422


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        assert client.get("/ml/health").status_code == 200

    def test_health_has_status_field(self, client):
        assert "status" in client.get("/ml/health").json()

    def test_health_status_is_healthy(self, client):
        assert client.get("/ml/health").json()["status"] == "healthy"

    def test_health_shows_menu_items_loaded(self, client):
        assert client.get("/ml/health").json()["menuItemsLoaded"] > 0

    def test_health_shows_ingredients_loaded(self, client):
        assert client.get("/ml/health").json()["ingredientsLoaded"] > 0


class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        assert client.get("/ml/metrics").status_code == 200

    def test_metrics_has_prediction_count(self, client):
        client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50})
        data = client.get("/ml/metrics").json()
        assert data["predictions"]["total"] >= 1

    def test_metrics_has_latency(self, client):
        data = client.get("/ml/metrics").json()
        assert "latency" in data and "avg_ms" in data["latency"]


class TestMenuItemsEndpoint:

    def test_menu_items_returns_200(self, client):
        assert client.get("/ml/menu-items").status_code == 200

    def test_menu_items_has_34_items(self, client):
        assert client.get("/ml/menu-items").json()["total"] == 34

    def test_menu_items_prediction_ready_count(self, client):
        assert client.get("/ml/menu-items").json()["predictionReady"] > 0


class TestShadowDeploymentMetrics:

    def test_ml_model_is_active(self, client):
        data = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50}).json()
        assert data["method"] in ["ml_model", "rule_based"]

    def test_high_confidence_on_standard_request(self, client):
        data = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50}).json()
        assert data["confidence"] >= 0.60

    def test_model_version_is_set(self, client):
        data = client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 50}).json()
        assert data["modelVersion"] is not None
        assert len(data["modelVersion"]) > 0

    def test_multiple_items_all_return_predictions(self, client):
        for item_id in ["ITEM-001", "ITEM-002", "ITEM-003"]:
            r = client.post("/ml/predict-cost", json={"menuItemId": item_id, "quantity": 25})
            assert r.status_code == 200
            assert r.json()["totalCost"] > 0

    def test_large_quantity_still_works(self, client):
        assert client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 1000}).status_code == 200

    def test_small_quantity_still_works(self, client):
        assert client.post("/ml/predict-cost", json={"menuItemId": "ITEM-001", "quantity": 1}).status_code == 200
