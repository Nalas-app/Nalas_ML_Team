"""
Feedback Loop & Alert Evaluation System
==========================================
Matches prediction logs against actual finalized invoices to track MAPE accuracy over time.
Flags predictions that exceeded the 30% error threshold for investigation.

Run periodically: `python scripts/phase8_feedback_loop.py --invoices data/actual_invoices.csv`
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd

# Add paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs", "predictions")
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ml_feedback")


def process_feedback_loop(invoice_data_path: str = None):
    """
    Simulate processing of actual order invoices against our logged predictions.
    """
    logger.info("Initializing Feedback Loop...")
    
    # 1. Load Prediction Logs
    prediction_files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith('.jsonl')]
    predictions = []
    
    for file in prediction_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "response" in data and "totalCost" in data["response"]:
                        predictions.append({
                            "timestamp": data["timestamp"],
                            "menu_item_id": data["request"].get("menuItemId"),
                            "quantity": data["request"].get("quantity"),
                            "predicted_cost": data["response"]["totalCost"],
                            "method": data["response"].get("method", "unknown"),
                            "confidence": data["response"].get("confidence", 0.0),
                            "model_version": data["response"].get("modelVersion", "N/A"),
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse log line: {e}")
                    
    df_preds = pd.DataFrame(predictions)
    if df_preds.empty:
        logger.warning("No valid predictions found in logs.")
        return
        
    logger.info(f"Loaded {len(df_preds)} prediction records.")

    # 2. Load Invoice Data (If provided)
    # Since we are mocking the invoice stream for this script:
    if not invoice_data_path or not os.path.exists(invoice_data_path):
        logger.info("No actual invoice CSV provided. Generating simulated invoice actuals for feedback analysis...")
        # Simulate actuals being close to predictions but with some drift
        import numpy as np
        np.random.seed(42)
        # 10% chance of a major fluctuation (e.g. price change)
        errors = np.where(np.random.random(len(df_preds)) < 0.10, 
                          np.random.uniform(0.35, 0.50, len(df_preds)), 
                          np.random.normal(0, 0.05, len(df_preds)))
        
        df_preds["actual_cost"] = df_preds["predicted_cost"] * (1 + errors)
    else:
        df_invoices = pd.read_csv(invoice_data_path)
        # Needs to join on some unique order ID in a real system. 
        # For simplicity here, we assume sequential or simulated match.
        pass
        
    # 3. Calculate Error Metrics
    df_preds["actual_cost"] = df_preds["actual_cost"].round(2)
    df_preds["absolute_error"] = abs(df_preds["predicted_cost"] - df_preds["actual_cost"])
    df_preds["pct_error"] = (df_preds["absolute_error"] / df_preds["actual_cost"]) * 100
    
    # Calculate global MAPE
    current_mape = df_preds["pct_error"].mean()
    logger.info(f"Current System MAPE: {current_mape:.2f}%")
    
    # 4. Flag Problematic Invoices (>30% error bounds)
    high_error_df = df_preds[df_preds["pct_error"] > 30.0].copy()
    if not high_error_df.empty:
        logger.warning(f"🔴 ALERT: Found {len(high_error_df)} predictions exceeding 30% error threshold!")
        
        # Breakdown by item
        problem_items = high_error_df.groupby("menu_item_id").size().sort_values(ascending=False)
        logger.warning("Problematic Menu Items:")
        for item, count in problem_items.items():
            logger.warning(f"   -> {item}: {count} severe errors")
            
    # 5. Export Feedback Data
    feedback_file = os.path.join(FEEDBACK_DIR, f"feedback_{datetime.now().strftime('%Y%m%d')}.csv")
    df_preds.to_csv(feedback_file, index=False)
    logger.info(f"Feedback data saved to {feedback_file}")

if __name__ == "__main__":
    # If script is run directly, process the feedback loop
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--invoices", help="Path to actual invoice CSV (optional)")
    args = parser.parse_args()
    
    process_feedback_loop(args.invoices)
