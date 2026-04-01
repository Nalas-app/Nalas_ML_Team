"""
Phase 4 — Feature Pipeline Tests
==================================
Tests for feature extraction, pipeline fit/transform, and serialization.
Run: python -m pytest tests/test_feature_pipeline.py -v
"""

import os
import sys
import pytest
import tempfile
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_service"))

from cost_engine import DataStore
from feature_pipeline import FeatureExtractor, FeaturePipeline, FeatureAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "structured")


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def data_store():
    ds = DataStore(DATA_DIR)
    ds.load()
    return ds


@pytest.fixture(scope="module")
def extractor(data_store):
    return FeatureExtractor(data_store)


@pytest.fixture(scope="module")
def sample_data():
    """Create a small sample DataFrame."""
    return pd.DataFrame([
        {'menu_item_id': 'ITEM-001', 'quantity': 25, 'event_date': '2026-11-20',
         'guest_count': 100, 'event_type': 'Wedding'},
        {'menu_item_id': 'ITEM-002', 'quantity': 10, 'event_date': '2026-07-15',
         'guest_count': 50, 'event_type': 'Birthday'},
        {'menu_item_id': 'ITEM-003', 'quantity': 5, 'event_date': '2026-03-10',
         'guest_count': 200, 'event_type': 'Corporate'},
    ])


@pytest.fixture(scope="module")
def fitted_pipeline(data_store, sample_data):
    pipeline = FeaturePipeline(data_store=data_store)
    pipeline.fit_transform(sample_data)
    return pipeline


# ============================================================
# FEATURE EXTRACTOR TESTS
# ============================================================

class TestFeatureExtractor:

    def test_base_ingredient_cost_positive(self, extractor):
        cost = extractor.calculate_base_ingredient_cost("ITEM-001")
        assert cost > 0

    def test_base_ingredient_cost_unknown(self, extractor):
        cost = extractor.calculate_base_ingredient_cost("UNKNOWN")
        assert cost == 0.0

    def test_recipe_complexity(self, extractor):
        complexity = extractor.calculate_recipe_complexity("ITEM-001")
        assert complexity > 0

    def test_menu_category(self, extractor):
        cat = extractor.get_menu_category("ITEM-001")
        assert cat == 'nonveg_main'

    def test_menu_category_unknown(self, extractor):
        cat = extractor.get_menu_category("UNKNOWN")
        assert cat == 'unknown'

    def test_extract_month(self):
        assert FeatureExtractor.extract_month("2026-11-20") == 11
        assert FeatureExtractor.extract_month("2026-01-05") == 1

    def test_extract_month_none(self):
        month = FeatureExtractor.extract_month(None)
        assert 1 <= month <= 12  # Current month

    def test_extract_day_of_week(self):
        dow = FeatureExtractor.extract_day_of_week("2026-11-20")
        assert 0 <= dow <= 6

    def test_is_weekend_saturday(self):
        # 2026-11-21 is Saturday
        assert FeatureExtractor.extract_is_weekend("2026-11-21") == 1

    def test_is_weekend_monday(self):
        # 2026-11-16 is Monday
        assert FeatureExtractor.extract_is_weekend("2026-11-16") == 0

    def test_is_wedding_season_november(self):
        assert FeatureExtractor.extract_is_wedding_season("2026-11-20") == 1

    def test_is_wedding_season_july(self):
        assert FeatureExtractor.extract_is_wedding_season("2026-07-15") == 0

    def test_is_festival_season_march(self):
        assert FeatureExtractor.extract_is_festival_season("2026-03-10") == 1

    def test_days_until_event(self):
        days = FeatureExtractor.calculate_days_until_event(
            "2026-12-25", "2026-12-01"
        )
        assert days == 24

    def test_days_until_event_none(self):
        days = FeatureExtractor.calculate_days_until_event(None)
        assert days == 30  # Default

    def test_historical_demand_no_data(self):
        demand = FeatureExtractor.calculate_historical_demand("ITEM-001")
        assert demand == 0.0

    def test_price_volatility_no_data(self):
        vol = FeatureExtractor.calculate_price_volatility(["ING-001"])
        assert vol == 0.0


# ============================================================
# FEATURE PIPELINE TESTS
# ============================================================

class TestFeaturePipeline:

    def test_extract_features_shape(self, data_store, sample_data):
        pipeline = FeaturePipeline(data_store=data_store)
        features = pipeline.extract_features(sample_data)
        assert features.shape[0] == 3  # 3 samples
        assert features.shape[1] == 16  # 16 raw features

    def test_extract_features_columns(self, data_store, sample_data):
        pipeline = FeaturePipeline(data_store=data_store)
        features = pipeline.extract_features(sample_data)
        expected_cols = [
            'base_ingredient_cost', 'quantity', 'recipe_complexity',
            'menu_category', 'has_perishable', 'month', 'day_of_week',
            'is_weekend', 'is_wedding_season', 'is_festival_season',
            'historical_demand', 'days_until_event', 'price_volatility',
            'recent_price_trend', 'guest_count', 'event_type'
        ]
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_fit_transform(self, fitted_pipeline):
        assert fitted_pipeline.is_fitted is True

    def test_transform_output_numeric(self, fitted_pipeline, sample_data):
        features = fitted_pipeline.extract_features(sample_data)
        X = fitted_pipeline.transform(features)
        # All columns should be numeric
        assert X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]

    def test_transform_no_nans(self, fitted_pipeline, sample_data):
        features = fitted_pipeline.extract_features(sample_data)
        X = fitted_pipeline.transform(features)
        assert X.isna().sum().sum() == 0

    def test_transform_single(self, fitted_pipeline):
        X = fitted_pipeline.transform_single(
            "ITEM-001", 25, "2026-11-20", 100, "Wedding"
        )
        assert X.shape[0] == 1
        assert X.shape[1] > 0
        assert X.isna().sum().sum() == 0

    def test_feature_names(self, fitted_pipeline):
        names = fitted_pipeline.feature_names
        assert len(names) > 0
        assert 'base_ingredient_cost' in names
        assert 'quantity' in names

    def test_save_and_load(self, fitted_pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_pipeline.pkl")
            fitted_pipeline.save(path)

            loaded = FeaturePipeline.load(path)
            assert loaded.is_fitted is True
            assert loaded.feature_names == fitted_pipeline.feature_names

    def test_loaded_pipeline_produces_same_output(self, fitted_pipeline, sample_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_pipeline.pkl")
            fitted_pipeline.save(path)
            loaded = FeaturePipeline.load(path)

            features = fitted_pipeline.extract_features(sample_data)
            X_original = fitted_pipeline.transform(features)
            X_loaded = loaded.transform(features)

            pd.testing.assert_frame_equal(X_original, X_loaded)


# ============================================================
# FEATURE ANALYZER TESTS
# ============================================================

class TestFeatureAnalyzer:

    def test_correlation_matrix(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'c': [5, 3, 1, 2, 4],
        })
        corr = FeatureAnalyzer.calculate_correlation_matrix(df)
        assert corr.shape == (3, 3)
        assert abs(corr.loc['a', 'b'] - 1.0) < 0.01  # Perfect correlation

    def test_find_high_correlations(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'c': [5, 3, 1, 2, 4],
        })
        corr = FeatureAnalyzer.calculate_correlation_matrix(df)
        high = FeatureAnalyzer.find_high_correlations(corr, threshold=0.9)
        assert len(high) >= 1
        assert high[0]['feature_1'] == 'a'
        assert high[0]['feature_2'] == 'b'

    def test_validate_top_features_pass(self):
        importance_df = pd.DataFrame({
            'feature': ['base_ingredient_cost', 'quantity', 'recipe_complexity',
                       'menu_category', 'guest_count'],
            'importance': [0.7, 0.15, 0.05, 0.05, 0.05],
            'importance_pct': [70, 15, 5, 5, 5],
            'cumulative_pct': [70, 85, 90, 95, 100],
            'rank': [1, 2, 3, 4, 5],
        })
        validations = FeatureAnalyzer.validate_top_features(importance_df)
        assert all(v['status'] == 'PASS' for v in validations)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
