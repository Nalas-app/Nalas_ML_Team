import sys
import os
import json
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml_service"))
from main import app, lifespan

def test_api():
    print("Testing ML API Integration...")
    # Need to run within lifespan context manually since TestClient doesn't fully trigger it the same way sometimes without explicitly saying so in newer Starlette
    with TestClient(app) as client:
        # 1. Health check
        resp = client.get("/ml/health")
        print("\n[Health]", resp.json())
        assert resp.status_code == 200
        assert resp.json()["modelLoaded"] is True
        
        # 2. Model info
        resp = client.get("/ml/model-info")
        print("\n[Model Info]", resp.json())
        assert resp.status_code == 200
        assert "ml_model" in resp.json()["method"]
        
        # 3. Predict cost (ML should handle this)
        payload = {
            "menuItemId": "ITEM-001",
            "quantity": 50,
            "eventDate": "2026-11-20",
            "guestCount": 150
        }
        resp = client.post("/ml/predict-cost", json=payload)
        print("\n[Predict - ML]", json.dumps(resp.json(), indent=2))
        assert resp.status_code == 200
        data = resp.json()
        assert data["totalCost"] > 0
        assert data["method"] == "ml_model"
        
        # 4. Fallback condition (Item lacking data)
        # We know from previous tests that fake items throw 400
        resp = client.post("/ml/predict-cost", json={"menuItemId": "FAKE", "quantity": 10})
        print("\n[Predict - Fake Item Error]", resp.status_code, resp.json())
        assert resp.status_code == 400

        print("\n✅ API Integration Test Passed!")

if __name__ == "__main__":
    test_api()
