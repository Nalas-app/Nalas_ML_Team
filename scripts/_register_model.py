"""Register existing model into the new model registry."""
import sys, os, pickle, json
sys.path.insert(0, 'ml_service')
from model_registry import ModelRegistry
from feature_pipeline import FeaturePipeline

MODEL_DIR = 'model'
registry = ModelRegistry(MODEL_DIR)

with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
    model = pickle.load(f)

pipeline = FeaturePipeline.load(os.path.join(MODEL_DIR, 'feature_pipeline.pkl'))

with open(os.path.join(MODEL_DIR, 'model_metadata.json')) as f:
    meta = json.load(f)

registry.register_model(
    model=model,
    pipeline=pipeline,
    version='v1.0.0',
    algorithm=meta.get('model_type', 'RandomForestRegressor'),
    training_samples=meta.get('training_samples', 620),
    features=meta.get('features', []),
    metrics=meta.get('metrics', {}),
    hyperparameters=meta.get('hyperparameters', {}),
    set_active=True,
)
print('Model v1.0.0 registered successfully')
print('Active:', registry.get_active_version())
print('Versions:', [v['version'] for v in registry.list_versions()])
