import sys
import os
import json
import logging

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml_service'))

from production_predictor import ProductionPredictor
from prediction_logger import PredictionLogger

def test_production_readiness():
    print("Initializing Predictor...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    data_dir = os.path.join(base_dir, "data", "structured")
    config_path = os.path.join(base_dir, "ml_service", "config", "business_rules.json")

    predictor = ProductionPredictor(model_dir, data_dir, config_path)
    predictor.initialize()

    logger = PredictionLogger()

    # 1. Test standard prediction
    print("\n[Test 1] Standard Prediction...")
    res1 = predictor.predict("ITEM-001", 50)
    logger.log_prediction({"menuItemId": "ITEM-001", "quantity": 50}, res1, 10.5)
    print(f"Result: {res1['totalCost']} (Outlier: {res1['_meta'].get('is_outlier', False)})")

    # 2. Test outlier prediction (extreme quantity or modified data)
    print("\n[Test 2] Potential Outlier Prediction (Large Qty)...")
    res2 = predictor.predict("ITEM-001", 1000)
    logger.log_prediction({"menuItemId": "ITEM-001", "quantity": 1000}, res2, 15.2)
    print(f"Result: {res2['totalCost']} (Outlier: {res2['_meta'].get('is_outlier', False)})")

    print(f"\nVerification Complete. Check logs/predictions/predictions.db")

if __name__ == "__main__":
    test_production_readiness()
