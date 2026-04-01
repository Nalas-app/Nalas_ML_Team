"""
Prediction Logger
==================
Logs every prediction to JSONL files for future ML training data.
"""

import json
import uuid
import os
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("ml_costing.logger")


class PredictionLogger:
    """Logs every prediction request/response for future model training."""

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base, "logs", "predictions")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._total_predictions = 0
        self._today_predictions = 0
        self._errors = 0
        self._latencies = []
        self._current_date = datetime.now().strftime("%Y-%m-%d")

    def log_prediction(self, request: dict, response: dict, latency_ms: float,
                       error: dict = None):
        """Log a prediction (success or failure)."""
        # Reset daily counter if date changed
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._today_predictions = 0
            self._current_date = today

        self._total_predictions += 1
        self._today_predictions += 1
        self._latencies.append(latency_ms)

        # Keep only last 1000 latencies
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]

        log_entry = {
            "prediction_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "request": request,
            "latency_ms": round(latency_ms, 2),
        }

        if error:
            log_entry["status"] = "error"
            log_entry["error"] = error
            self._errors += 1
        else:
            log_entry["status"] = "success"
            # Extract key response fields (without _meta)
            log_entry["response"] = {
                k: v for k, v in response.items() if not k.startswith("_")
            }
            # Store meta separately for training data
            if "_meta" in response:
                log_entry["meta"] = response["_meta"]

        # Write to daily log file
        log_file = self.log_dir / f"predictions_{today}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write prediction log: {e}")

    def get_metrics(self) -> dict:
        """Return current metrics."""
        import numpy as np

        latencies = self._latencies if self._latencies else [0]
        lat_arr = np.array(latencies)

        return {
            "total_predictions": self._total_predictions,
            "predictions_today": self._today_predictions,
            "avg_latency_ms": round(float(np.mean(lat_arr)), 2),
            "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2) if len(lat_arr) > 1 else 0,
            "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 2) if len(lat_arr) > 1 else 0,
            "error_count": self._errors,
            "error_rate_pct": round(self._errors / max(self._total_predictions, 1) * 100, 2),
        }
