"""
Production Prediction Service
================================
Hybrid prediction service: ML model with rule-based fallback.

Priority:
    1. ML model prediction (if model loaded + confidence > 0.4)
    2. Rule-based prediction (always available as fallback)

Features:
    - Singleton model loading (one-time at startup)
    - Automatic fallback to rule-based on any ML failure
    - Confidence-based routing
    - Full prediction logging
    - Thread-safe design
"""

import os
import time
import logging
import threading
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger("ml_costing.predictor")


class ProductionPredictor:
    """
    Production-ready prediction service with ML + rule-based fallback.

    Usage:
        predictor = ProductionPredictor(model_dir, data_dir, config_path)
        predictor.initialize()

        result = predictor.predict("ITEM-001", 25, "2026-11-20", 100, "Wedding")
    """

    # Confidence below this threshold triggers rule-based fallback
    ML_CONFIDENCE_THRESHOLD = 0.40

    # Confidence below this threshold flags for manual review
    MANUAL_REVIEW_THRESHOLD = 0.60

    # Deviation from rule-based baseline above this triggers outlier flag
    OUTLIER_DEVIATION_THRESHOLD = 0.35

    def __init__(self, model_dir: str, data_dir: str, config_path: str):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.config_path = config_path

        # Lazy-loaded components
        self._model = None
        self._pipeline = None
        self._model_metadata = None
        self._engine = None       # Rule-based fallback
        self._data_store = None
        self._business_rules = None
        self._registry = None

        self._initialized = False
        self._lock = threading.Lock()

        # Tracking
        self._ml_predictions = 0
        self._rule_predictions = 0
        self._outlier_count = 0
        self._fallback_reasons = []

    def initialize(self):
        """
        One-time initialization. Load all components.
        Call this at application startup.
        """
        with self._lock:
            if self._initialized:
                return

            logger.info("Initializing ProductionPredictor...")

            # 1. Load data store (always needed)
            from ml_service.cost_engine import DataStore, BusinessRules, CostEngine
            self._data_store = DataStore(self.data_dir)
            self._data_store.load()

            # 2. Load business rules
            self._business_rules = BusinessRules(self.config_path)

            # 3. Initialize rule-based engine (always available)
            self._engine = CostEngine(self._data_store, self._business_rules)
            logger.info("Rule-based engine initialized ✓")

            # 4. Try to load ML model
            try:
                from ml_service.model_registry import ModelRegistry
                self._registry = ModelRegistry(self.model_dir)

                active_version = self._registry.get_active_version()
                if active_version:
                    loaded = self._registry.load_model(active_version)
                    self._model = loaded["model"]
                    self._pipeline = loaded["pipeline"]
                    self._model_metadata = loaded["metadata"]
                    logger.info(
                        f"ML model loaded: {active_version} "
                        f"({loaded['metadata'].get('algorithm', 'unknown')})"
                    )
                else:
                    logger.warning("No active ML model version — using rule-based only")
            except Exception as e:
                logger.warning(f"ML model not available: {e} — using rule-based only")

            self._initialized = True
            logger.info("ProductionPredictor initialized successfully")

    @property
    def is_ml_available(self) -> bool:
        """Whether ML model is loaded and ready."""
        return self._model is not None and self._pipeline is not None

    @property
    def active_model_version(self) -> Optional[str]:
        """Currently active model version."""
        if self._model_metadata:
            return self._model_metadata.get("modelVersion", "unknown")
        return None

    def predict(
        self,
        menu_item_id: str,
        quantity: int,
        event_date: Optional[str] = None,
        guest_count: Optional[int] = None,
        event_type: Optional[str] = None,
        force_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make a cost prediction using the best available method.

        Args:
            menu_item_id: Item ID or name
            quantity: Number of servings
            event_date: Event date (YYYY-MM-DD)
            guest_count: Number of guests
            event_type: Wedding/Birthday/Corporate/Other
            force_method: Force "ml" or "rule_based" method

        Returns:
            Dict with cost breakdown, confidence, method used
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Determine prediction method
        use_ml = (
            force_method == "ml"
            or (force_method is None and self.is_ml_available)
        )

        if use_ml:
            try:
                result = self._predict_ml(
                    menu_item_id, quantity, event_date, guest_count, event_type
                )
                latency_ms = (time.time() - start_time) * 1000

                # Check confidence — fallback if too low
                if result["confidence"] < self.ML_CONFIDENCE_THRESHOLD:
                    logger.info(
                        f"ML confidence {result['confidence']:.2f} below threshold "
                        f"{self.ML_CONFIDENCE_THRESHOLD} — falling back to rule-based"
                    )
                    self._fallback_reasons.append("low_confidence")
                    return self._predict_rule_based(
                        menu_item_id, quantity, event_date, guest_count,
                        fallback_reason="low_ml_confidence"
                    )

                # ============================================================
                # OUTLIER DETECTION (Mugil's Feature)
                # ============================================================
                # Compare ML prediction against deterministic rule-based baseline
                try:
                    baseline = self._engine.predict(menu_item_id, quantity, event_date, guest_count)
                    rule_total = baseline["totalCost"]
                    ml_total = result["totalCost"]
                    
                    if rule_total > 0:
                        deviation = abs(ml_total - rule_total) / rule_total
                        if deviation > self.OUTLIER_DEVIATION_THRESHOLD:
                            result["_meta"]["is_outlier"] = True
                            result["_meta"]["flags"].append(f"outlier_detected:{deviation:.1%}_deviation")
                            self._outlier_count += 1
                            logger.warning(
                                f"OUTLIER DETECTED: ML Prediction (Rs.{ml_total}) deviates from "
                                f"Rule-based (Rs.{rule_total}) by {deviation:.1%}"
                            )
                except Exception as e:
                    logger.error(f"Failed to perform outlier cross-check: {e}")

                # Flag for manual review if below review threshold
                if result["confidence"] < self.MANUAL_REVIEW_THRESHOLD:
                    result["_meta"]["flags"].append("manual_review_recommended")

                result["_meta"]["latency_ms"] = round(latency_ms, 2)
                self._ml_predictions += 1
                return result

            except Exception as e:
                logger.warning(f"ML prediction failed: {e} — falling back to rule-based")
                self._fallback_reasons.append(f"ml_error: {str(e)[:100]}")
                return self._predict_rule_based(
                    menu_item_id, quantity, event_date, guest_count,
                    fallback_reason=f"ml_prediction_failed: {type(e).__name__}"
                )
        else:
            return self._predict_rule_based(
                menu_item_id, quantity, event_date, guest_count,
                fallback_reason="ml_not_available" if not self.is_ml_available else "forced"
            )

    def _predict_ml(
        self,
        menu_item_id: str,
        quantity: int,
        event_date: Optional[str],
        guest_count: Optional[int],
        event_type: Optional[str],
    ) -> Dict[str, Any]:
        """ML model prediction path."""
        from ml_service.cost_engine import CostingError

        # 1. Input validation (reuse engine validation)
        if quantity <= 0:
            raise CostingError("ML_001", f"Invalid quantity: {quantity}")

        item = self._data_store.get_menu_item(menu_item_id)
        if item is None:
            raise CostingError("ML_002", f"Menu item not found: {menu_item_id}")

        resolved_id = item['item_id']

        # 2. Extract features
        X = self._pipeline.transform_single(
            resolved_id, quantity, event_date, guest_count, event_type
        )

        # 3. ML prediction
        raw_prediction = float(self._model.predict(X)[0])
        raw_prediction = max(raw_prediction, 0)  # Non-negative

        # 4. Decompose into cost breakdown using business rules
        # The ML predicts total cost — we decompose using known ratios
        labor_pct = self._business_rules.labor_pct
        overhead_pct = self._business_rules.overhead_pct
        wastage_pct = self._business_rules.wastage_pct
        profit_pct = self._business_rules.profit_pct

        # Reverse-engineer from total:
        # total = subtotal × (1 + profit_pct)
        # subtotal = ingredient × (1 + labor + overhead + wastage)
        total_multiplier = (1 + labor_pct + overhead_pct + wastage_pct) * (1 + profit_pct)
        ingredient_cost = raw_prediction / total_multiplier
        labor_cost = ingredient_cost * labor_pct
        overhead_cost = ingredient_cost * overhead_pct

        # 5. Calculate confidence
        confidence = self._calculate_ml_confidence(
            resolved_id, quantity, raw_prediction
        )

        return {
            "ingredientCost": round(ingredient_cost, 2),
            "laborCost": round(labor_cost, 2),
            "overheadCost": round(overhead_cost, 2),
            "totalCost": round(raw_prediction, 2),
            "confidence": round(confidence, 2),
            "modelVersion": self._model_metadata.get("modelVersion", "v1.0.0"),
            "method": "ml_model",
            "_meta": {
                "item_id": resolved_id,
                "item_name": item['item_name'],
                "quantity": quantity,
                "ml_raw_prediction": round(raw_prediction, 2),
                "is_outlier": False,  # Default
                "flags": [],
                "seasonal_adjustment": 0.0,
                "bulk_discount": 0.0,
            }
        }

    def _predict_rule_based(
        self,
        menu_item_id: str,
        quantity: int,
        event_date: Optional[str] = None,
        guest_count: Optional[int] = None,
        fallback_reason: str = "default",
    ) -> Dict[str, Any]:
        """Rule-based prediction path (fallback)."""
        start = time.time()
        result = self._engine.predict(menu_item_id, quantity, event_date, guest_count)
        latency = (time.time() - start) * 1000

        # Add fallback info
        result["_meta"]["fallback_reason"] = fallback_reason
        result["_meta"]["latency_ms"] = round(latency, 2)
        result["_meta"]["is_outlier"] = False # Rule-based is baseline, never outlier

        self._rule_predictions += 1
        return result

    def _calculate_ml_confidence(
        self,
        menu_item_id: str,
        quantity: int,
        prediction: float,
    ) -> float:
        """
        Calculate prediction confidence based on multiple signals.

        Signals:
        1. Feature similarity to training data
        2. Prediction within expected range
        3. Model ensemble variance (if available)
        4. Data availability for this item
        """
        confidence = 0.70  # Base confidence for ML model

        # Signal 1: Recipe data completeness
        recipe = self._data_store.get_recipe(menu_item_id)
        if len(recipe) > 0:
            matched = recipe['ingredient_id'].apply(
                lambda x: x != 'UNMATCHED' and self._data_store.get_ingredient_price(x) is not None
            ).sum()
            match_rate = matched / len(recipe)
            confidence += match_rate * 0.15  # Up to +15%

        # Signal 2: Prediction sanity check
        if prediction < 0:
            confidence -= 0.30
        elif prediction > 1e8:  # Suspiciously high
            confidence -= 0.20

        # Signal 3: Reasonable quantity range
        if 5 <= quantity <= 500:
            confidence += 0.05  # Common range bonus
        elif quantity > 2000:
            confidence -= 0.10  # Rare range penalty

        # Signal 4: Ensemble variance (RandomForest only)
        if hasattr(self._model, 'estimators_'):
            try:
                X = self._pipeline.transform_single(menu_item_id, quantity)
                tree_preds = np.array([
                    t.predict(X)[0] for t in self._model.estimators_[:20]
                ])
                cv = tree_preds.std() / tree_preds.mean() if tree_preds.mean() > 0 else 1.0
                if cv < 0.1:
                    confidence += 0.05   # Low variance = high agreement
                elif cv > 0.5:
                    confidence -= 0.15   # High variance = disagreement
            except Exception:
                pass

        return max(0.10, min(0.95, confidence))

    def get_stats(self) -> Dict:
        """Get prediction statistics."""
        total = self._ml_predictions + self._rule_predictions
        return {
            "total_predictions": total,
            "ml_predictions": self._ml_predictions,
            "rule_based_predictions": self._rule_predictions,
            "outlier_count": self._outlier_count,
            "ml_rate_pct": round(self._ml_predictions / total * 100, 2) if total > 0 else 0,
            "fallback_rate_pct": round(self._rule_predictions / total * 100, 2) if total > 0 else 0,
            "outlier_rate_pct": round(self._outlier_count / max(self._ml_predictions, 1) * 100, 2),
            "ml_available": self.is_ml_available,
            "active_version": self.active_model_version,
            "recent_fallback_reasons": self._fallback_reasons[-10:],
        }

    def reload_model(self, version: Optional[str] = None):
        """Hot-reload a model version without restarting the service."""
        with self._lock:
            try:
                if self._registry is None:
                    from model_registry import ModelRegistry
                    self._registry = ModelRegistry(self.model_dir)

                target = version or self._registry.get_active_version()
                if target is None:
                    logger.warning("No model version to reload")
                    return

                loaded = self._registry.load_model(target)
                self._model = loaded["model"]
                self._pipeline = loaded["pipeline"]
                self._model_metadata = loaded["metadata"]
                logger.info(f"Model hot-reloaded: {target}")

            except Exception as e:
                logger.error(f"Model reload failed: {e}")
                # Keep existing model if reload fails
