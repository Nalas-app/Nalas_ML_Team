"""
Model Registry & Versioning
=============================
Manages model versions, metadata, serialization, and rollback.

Usage:
    registry = ModelRegistry("model/")
    registry.register_model(model, pipeline, metadata, version="v1.0.0")
    loaded = registry.load_model("v1.0.0")
    registry.rollback("v0.9.0")
"""

import os
import json
import pickle
import shutil
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger("ml_costing.registry")


class ModelRegistry:
    """
    Manages ML model versions with metadata, performance tracking, and rollback.

    Directory structure:
        model/
        ├── registry.json         ← Version history & current active version
        ├── v1.0.0/
        │   ├── model.pkl         ← Serialized model
        │   ├── pipeline.pkl      ← Feature pipeline
        │   └── metadata.json     ← Training info, metrics, limitations
        ├── v1.1.0/
        │   ├── model.pkl
        │   ├── pipeline.pkl
        │   └── metadata.json
        └── active -> v1.0.0/     ← Symlink/copy to active version
    """

    REGISTRY_FILE = "registry.json"

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self._registry_path = os.path.join(model_dir, self.REGISTRY_FILE)
        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load or create registry file."""
        if os.path.exists(self._registry_path):
            with open(self._registry_path, 'r') as f:
                return json.load(f)
        return {
            "active_version": None,
            "versions": [],
            "created_at": datetime.now().isoformat(),
        }

    def _save_registry(self):
        """Persist registry to disk."""
        self._registry["updated_at"] = datetime.now().isoformat()
        with open(self._registry_path, 'w') as f:
            json.dump(self._registry, f, indent=2, default=str)

    def register_model(
        self,
        model: Any,
        pipeline: Any,
        version: str,
        algorithm: str = "GradientBoosting",
        training_samples: int = 0,
        features: Optional[List[str]] = None,
        metrics: Optional[Dict] = None,
        hyperparameters: Optional[Dict] = None,
        limitations: Optional[List[str]] = None,
        business_rules: Optional[Dict] = None,
        set_active: bool = True,
    ):
        """
        Register a new model version.

        Args:
            model: Trained sklearn model
            pipeline: Fitted FeaturePipeline
            version: Semantic version string (e.g., "v1.0.0")
            algorithm: Algorithm name
            training_samples: Number of training samples
            features: List of feature names
            metrics: Dict of performance metrics (mape, rmse, r2)
            hyperparameters: Model hyperparameters
            limitations: Known limitations
            business_rules: Business rule configuration
            set_active: Whether to set this as the active version
        """
        version_dir = os.path.join(self.model_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(version_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save pipeline
        pipeline_path = os.path.join(version_dir, "pipeline.pkl")
        if hasattr(pipeline, 'save'):
            pipeline.save(pipeline_path)
        else:
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline, f)

        # Create metadata
        metadata = {
            "modelVersion": version,
            "algorithm": algorithm,
            "trainedDate": datetime.now().isoformat(),
            "trainingDataSize": training_samples,
            "features": features or [],
            "performanceMetrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "businessRules": business_rules or {
                "laborMultiplier": 0.15,
                "overheadMultiplier": 0.10,
                "wastageMultiplier": 0.05,
                "profitMultiplier": 0.10,
            },
            "limitations": limitations or [
                "Accuracy degrades for items with < 10 historical orders",
                "Assumes current ingredient prices",
                "Does not handle custom menu items",
                "Seasonality only captured if event_date provided",
            ],
        }

        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update registry
        version_entry = {
            "version": version,
            "algorithm": algorithm,
            "registered_at": datetime.now().isoformat(),
            "training_samples": training_samples,
            "metrics": metrics or {},
            "status": "active" if set_active else "registered",
        }

        # Remove duplicate version if re-registering
        self._registry["versions"] = [
            v for v in self._registry["versions"] if v["version"] != version
        ]
        self._registry["versions"].append(version_entry)

        if set_active:
            self._set_active(version)

        self._save_registry()
        logger.info(f"Model {version} registered successfully")

    def _set_active(self, version: str):
        """Set a version as the active production model."""
        old_active = self._registry.get("active_version")
        self._registry["active_version"] = version

        # Update status
        for v in self._registry["versions"]:
            if v["version"] == version:
                v["status"] = "active"
            elif v["version"] == old_active:
                v["status"] = "inactive"

        # Copy to active directory
        version_dir = os.path.join(self.model_dir, version)
        active_dir = os.path.join(self.model_dir, "active")

        if os.path.exists(active_dir):
            shutil.rmtree(active_dir)
        shutil.copytree(version_dir, active_dir)

        logger.info(f"Active model set to {version} (was {old_active})")

    def load_model(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a model by version. If no version specified, loads active model.

        Returns:
            Dict with keys: model, pipeline, metadata, version
        """
        if version is None:
            version = self._registry.get("active_version")
        if version is None:
            raise FileNotFoundError("No active model version set")

        version_dir = os.path.join(self.model_dir, version)
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"Model version {version} not found")

        # Load model
        model_path = os.path.join(version_dir, "model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load pipeline
        pipeline_path = os.path.join(version_dir, "pipeline.pkl")
        try:
            from ml_service.feature_pipeline import FeaturePipeline
            pipeline = FeaturePipeline.load(pipeline_path)
        except Exception:
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)

        # Load metadata
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return {
            "model": model,
            "pipeline": pipeline,
            "metadata": metadata,
            "version": version,
        }

    def rollback(self, version: str):
        """Rollback to a previous model version."""
        if not any(v["version"] == version for v in self._registry["versions"]):
            raise ValueError(f"Version {version} not found in registry")

        old_version = self._registry.get("active_version")
        self._set_active(version)
        self._save_registry()

        logger.warning(f"Rolled back from {old_version} to {version}")

    def get_active_version(self) -> Optional[str]:
        """Get the current active model version."""
        return self._registry.get("active_version")

    def list_versions(self) -> List[Dict]:
        """List all registered model versions."""
        return self._registry.get("versions", [])

    def compare_versions(self, v1: str, v2: str) -> Dict:
        """Compare metrics between two versions."""
        versions = {v["version"]: v for v in self._registry["versions"]}
        if v1 not in versions or v2 not in versions:
            raise ValueError(f"Version not found: {v1} or {v2}")

        m1 = versions[v1].get("metrics", {})
        m2 = versions[v2].get("metrics", {})

        comparison = {}
        all_metrics = set(list(m1.keys()) + list(m2.keys()))
        for metric in all_metrics:
            val1 = m1.get(metric, "N/A")
            val2 = m2.get(metric, "N/A")
            comparison[metric] = {
                v1: val1,
                v2: val2,
                "better": v1 if (isinstance(val1, (int, float)) and isinstance(val2, (int, float))
                                  and val1 < val2 and metric in ('mape', 'rmse', 'mae'))
                           else (v2 if isinstance(val1, (int, float)) and isinstance(val2, (int, float))
                                      and val2 < val1 and metric in ('mape', 'rmse', 'mae')
                                 else "tie"),
            }

        return comparison

    def should_deploy(self, new_metrics: Dict, threshold_pct: float = 2.0) -> Dict:
        """
        Decision logic: should a new model be deployed?

        Rules:
        - If new MAPE < current MAPE: Deploy (improvement)
        - If new MAPE > current MAPE + threshold: Do NOT deploy (degradation)
        - If similar performance: Deploy (benefits from newer data)
        """
        active = self._registry.get("active_version")
        if active is None:
            return {"deploy": True, "reason": "No active model — deploy first version"}

        current_metrics = {}
        for v in self._registry["versions"]:
            if v["version"] == active:
                current_metrics = v.get("metrics", {})
                break

        new_mape = new_metrics.get("mape", 100)
        current_mape = current_metrics.get("mape", 100)

        if new_mape < current_mape:
            return {
                "deploy": True,
                "reason": f"Improvement: MAPE {current_mape:.2f}% → {new_mape:.2f}%",
            }
        elif new_mape > current_mape + threshold_pct:
            return {
                "deploy": False,
                "reason": (f"Degradation: MAPE {current_mape:.2f}% → {new_mape:.2f}% "
                          f"(exceeds {threshold_pct}% threshold)"),
            }
        else:
            return {
                "deploy": True,
                "reason": f"Similar performance ({new_mape:.2f}% vs {current_mape:.2f}%) — deploy for newer data",
            }
