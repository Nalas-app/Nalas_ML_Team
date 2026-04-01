"""
Feature Engineering Pipeline
==============================
Phase 4: ML Feature Engineering for Nalas ML Costing Engine

This module provides:
    1. Feature extraction functions for each of the 16 features
    2. FeaturePipeline class for fit/transform workflow
    3. Feature preprocessing (scaling, encoding, imputation)
    4. Pipeline serialization (save/load with model)

Usage:
    # Training
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(train_data)
    pipeline.save("model/v1.0.0/feature_pipeline.pkl")

    # Prediction
    pipeline = FeaturePipeline.load("model/v1.0.0/feature_pipeline.pkl")
    X_new = pipeline.transform(new_data)
"""

import os
import json
import pickle
import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np

logger = logging.getLogger("ml_costing.features")


# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

class FeatureExtractor:
    """
    Extracts raw features from input data.
    Each method corresponds to one feature from the design document.
    """

    # Wedding season months (Oct-Feb)
    WEDDING_MONTHS = {10, 11, 12, 1, 2}
    # Festival season months (Mar-Apr)
    FESTIVAL_MONTHS = {3, 4}

    # Category mapping for menu items
    CATEGORY_MAP = {
        'Non-Veg Main Course': 'nonveg_main',
        'Veg Main Course': 'veg_main',
        'Rice': 'rice',
        'Breakfast': 'breakfast',
        'Side Dish': 'side_dish',
        'Dessert': 'dessert',
        'Starter': 'starter',
        'Beverage': 'beverage',
    }

    def __init__(self, data_store=None):
        """
        Args:
            data_store: DataStore instance for recipe/ingredient lookups.
                        None if features are pre-computed in the dataframe.
        """
        self.data_store = data_store

    # ── BASE FEATURES ──────────────────────────────────────

    def calculate_base_ingredient_cost(self, menu_item_id: str) -> float:
        """
        Feature 1: Total ingredient cost from recipe.
        Primary cost driver — ingredients are 60-70% of total cost.
        """
        if self.data_store is None:
            return 0.0

        recipe = self.data_store.get_recipe(menu_item_id)
        if len(recipe) == 0:
            return 0.0

        total_cost = 0.0
        for _, row in recipe.iterrows():
            ing_id = row.get('ingredient_id', 'UNMATCHED')
            qty = row.get('quantity_per_base_unit', None)

            if pd.isna(qty) or ing_id == 'UNMATCHED':
                continue

            price = self.data_store.get_ingredient_price(ing_id)
            if price and price > 0:
                total_cost += qty * price

        return round(total_cost, 2)

    def calculate_recipe_complexity(self, menu_item_id: str) -> int:
        """
        Feature 3: Number of ingredients in recipe.
        More ingredients = generally higher cost and labor.
        """
        if self.data_store is None:
            return 0

        recipe = self.data_store.get_recipe(menu_item_id)
        return len(recipe)

    def get_menu_category(self, menu_item_id: str) -> str:
        """
        Feature 4: Menu item category.
        Non-veg costs more than veg; desserts differ from mains.
        """
        if self.data_store is None:
            return 'unknown'

        item = self.data_store.get_menu_item(menu_item_id)
        if item is None:
            return 'unknown'

        raw_cat = item.get('category', 'unknown')
        return self.CATEGORY_MAP.get(raw_cat, 'other')

    def calculate_has_perishable(self, menu_item_id: str) -> int:
        """
        Feature 5: Whether recipe contains perishable ingredients.
        Perishable ingredients (meat, dairy) cost more, higher wastage.
        """
        if self.data_store is None:
            return 0

        recipe = self.data_store.get_recipe(menu_item_id)
        for _, row in recipe.iterrows():
            ing_id = row.get('ingredient_id', 'UNMATCHED')
            if ing_id == 'UNMATCHED':
                continue

            match = self.data_store.ingredients[
                self.data_store.ingredients['ingredient_id'] == ing_id
            ]
            if len(match) > 0 and match.iloc[0].get('is_perishable', False):
                return 1

        return 0

    # ── TIME-BASED FEATURES ────────────────────────────────

    @staticmethod
    def extract_month(event_date: Optional[str]) -> int:
        """Feature 6: Month of the event (1-12)."""
        if event_date is None:
            return datetime.now().month
        try:
            return date.fromisoformat(str(event_date)).month
        except (ValueError, TypeError):
            return datetime.now().month

    @staticmethod
    def extract_day_of_week(event_date: Optional[str]) -> int:
        """Feature 7: Day of week (0=Monday, 6=Sunday)."""
        if event_date is None:
            return datetime.now().weekday()
        try:
            return date.fromisoformat(str(event_date)).weekday()
        except (ValueError, TypeError):
            return datetime.now().weekday()

    @staticmethod
    def extract_is_weekend(event_date: Optional[str]) -> int:
        """Feature 8: Whether the event is on weekend (Sat=5, Sun=6)."""
        if event_date is None:
            return 0
        try:
            dow = date.fromisoformat(str(event_date)).weekday()
            return 1 if dow >= 5 else 0
        except (ValueError, TypeError):
            return 0

    @classmethod
    def extract_is_wedding_season(cls, event_date: Optional[str]) -> int:
        """Feature 9: Whether event is during wedding season (Oct-Feb)."""
        month = cls.extract_month(event_date)
        return 1 if month in cls.WEDDING_MONTHS else 0

    @classmethod
    def extract_is_festival_season(cls, event_date: Optional[str]) -> int:
        """Feature 10: Whether event is during festival season (Mar-Apr)."""
        month = cls.extract_month(event_date)
        return 1 if month in cls.FESTIVAL_MONTHS else 0

    # ── DEMAND FEATURES ────────────────────────────────────

    @staticmethod
    def calculate_historical_demand(
        menu_item_id: str,
        order_history: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Feature 11: Average monthly orders for this item.
        Requires historical order data. Returns 0 if unavailable.
        """
        if order_history is None or len(order_history) == 0:
            return 0.0

        item_orders = order_history[
            order_history['menu_item_id'] == menu_item_id
        ]
        if len(item_orders) == 0:
            return 0.0

        # Calculate monthly average
        if 'created_at' in item_orders.columns:
            item_orders = item_orders.copy()
            item_orders['month'] = pd.to_datetime(
                item_orders['created_at']
            ).dt.to_period('M')
            monthly_counts = item_orders.groupby('month').size()
            return round(float(monthly_counts.mean()), 2)

        return float(len(item_orders))

    @staticmethod
    def calculate_days_until_event(
        event_date: Optional[str],
        order_date: Optional[str] = None
    ) -> int:
        """
        Feature 12: Days between order placement and event.
        Captures booking lead time. Longer lead = better pricing.
        """
        if event_date is None:
            return 30  # Default: 30 days lead time

        try:
            evt = date.fromisoformat(str(event_date))
            if order_date:
                ord_dt = date.fromisoformat(str(order_date))
            else:
                ord_dt = date.today()
            delta = (evt - ord_dt).days
            return max(delta, 0)  # No negative days
        except (ValueError, TypeError):
            return 30

    # ── PRICE FEATURES ─────────────────────────────────────

    @staticmethod
    def calculate_price_volatility(
        ingredient_ids: List[str],
        price_history: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Feature 13: Average price volatility across recipe ingredients.
        Uses coefficient of variation of historical prices.
        Returns 0 if no history available.
        """
        if price_history is None or len(price_history) == 0:
            return 0.0

        volatilities = []
        for ing_id in ingredient_ids:
            ing_prices = price_history[
                price_history['ingredient_id'] == ing_id
            ]['new_price']

            if len(ing_prices) > 1:
                cv = ing_prices.std() / ing_prices.mean() if ing_prices.mean() > 0 else 0
                volatilities.append(cv)

        return round(float(np.mean(volatilities)), 4) if volatilities else 0.0

    @staticmethod
    def calculate_recent_price_trend(
        ingredient_ids: List[str],
        price_history: Optional[pd.DataFrame] = None,
        days: int = 30
    ) -> float:
        """
        Feature 14: Average price change over last N days.
        Positive = prices rising, Negative = prices falling.
        Returns 0 if no history available.
        """
        if price_history is None or len(price_history) == 0:
            return 0.0

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        recent = price_history[
            pd.to_datetime(price_history['changed_at']) >= cutoff
        ]

        if len(recent) == 0:
            return 0.0

        trends = []
        for ing_id in ingredient_ids:
            ing_changes = recent[recent['ingredient_id'] == ing_id]
            if len(ing_changes) > 0:
                pct_changes = ing_changes['change_pct'].dropna()
                if len(pct_changes) > 0:
                    trends.append(float(pct_changes.mean()))

        return round(float(np.mean(trends)), 4) if trends else 0.0


# ============================================================
# FEATURE PIPELINE
# ============================================================

class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.

    Handles:
    - Feature extraction from raw inputs
    - Numeric feature scaling (StandardScaler)
    - Categorical feature encoding (one-hot)
    - Missing value imputation
    - Serialization (save/load with model)

    Usage:
        # Training
        pipeline = FeaturePipeline(data_store=data_store)
        X_train = pipeline.fit_transform(train_df)
        pipeline.save("pipeline.pkl")

        # Prediction
        pipeline = FeaturePipeline.load("pipeline.pkl")
        X = pipeline.transform(new_df)
    """

    # Features that need scaling
    NUMERIC_FEATURES = [
        'base_ingredient_cost',
        'quantity',
        'recipe_complexity',
        'guest_count',
        'historical_demand',
        'days_until_event',
        'price_volatility',
        'recent_price_trend',
    ]

    # Features that need one-hot encoding
    CATEGORICAL_FEATURES = [
        'menu_category',
        'event_type',
    ]

    # Binary features (no transformation needed)
    BINARY_FEATURES = [
        'has_perishable',
        'is_weekend',
        'is_wedding_season',
        'is_festival_season',
    ]

    # Cyclical feature
    CYCLICAL_FEATURES = [
        'month',        # Encoded as sin/cos
        'day_of_week',  # Encoded as sin/cos
    ]

    def __init__(self, data_store=None):
        self.data_store = data_store
        self.extractor = FeatureExtractor(data_store)

        # Scaling parameters (fitted during training)
        self._numeric_means = {}
        self._numeric_stds = {}

        # Encoding parameters
        self._category_values = {}  # Known category values per feature

        # Imputation defaults
        self._imputation_values = {}

        self._is_fitted = False
        self._feature_names = []

    def extract_features(
        self,
        data: pd.DataFrame,
        order_history: Optional[pd.DataFrame] = None,
        price_history: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract all 16 raw features from input data.

        Args:
            data: DataFrame with columns: menu_item_id, quantity, event_date,
                  guest_count, event_type
            order_history: Optional historical orders for demand features
            price_history: Optional price change history for price features

        Returns:
            DataFrame with all 16 features as columns
        """
        features = pd.DataFrame(index=data.index)

        # ── Base features ──
        if self.data_store:
            features['base_ingredient_cost'] = data['menu_item_id'].apply(
                self.extractor.calculate_base_ingredient_cost
            )
            features['recipe_complexity'] = data['menu_item_id'].apply(
                self.extractor.calculate_recipe_complexity
            )
            features['menu_category'] = data['menu_item_id'].apply(
                self.extractor.get_menu_category
            )
            features['has_perishable'] = data['menu_item_id'].apply(
                self.extractor.calculate_has_perishable
            )
        else:
            # Features may already be in the dataframe
            for col in ['base_ingredient_cost', 'recipe_complexity',
                        'menu_category', 'has_perishable']:
                if col in data.columns:
                    features[col] = data[col]
                else:
                    features[col] = 0

        features['quantity'] = data['quantity'].astype(float)

        # ── Time features ──
        event_dates = data.get('event_date', pd.Series([None] * len(data)))
        features['month'] = event_dates.apply(self.extractor.extract_month)
        features['day_of_week'] = event_dates.apply(self.extractor.extract_day_of_week)
        features['is_weekend'] = event_dates.apply(self.extractor.extract_is_weekend)
        features['is_wedding_season'] = event_dates.apply(
            self.extractor.extract_is_wedding_season
        )
        features['is_festival_season'] = event_dates.apply(
            self.extractor.extract_is_festival_season
        )

        # ── Demand features ──
        features['historical_demand'] = data['menu_item_id'].apply(
            lambda x: self.extractor.calculate_historical_demand(x, order_history)
        )
        features['days_until_event'] = event_dates.apply(
            lambda x: self.extractor.calculate_days_until_event(x)
        )

        # ── Price features ──
        features['price_volatility'] = 0.0  # Requires price history
        features['recent_price_trend'] = 0.0  # Requires price history

        # ── Contextual features ──
        features['guest_count'] = data.get('guest_count', pd.Series([50] * len(data))).fillna(50).astype(float)
        features['event_type'] = data.get('event_type', pd.Series(['Other'] * len(data))).fillna('Other')

        return features

    def fit(self, features: pd.DataFrame) -> 'FeaturePipeline':
        """
        Learn scaling/encoding parameters from training data.
        Must be called before transform().
        """
        logger.info(f"Fitting pipeline on {len(features)} samples...")

        # 1. Learn numeric scaling parameters
        for col in self.NUMERIC_FEATURES:
            if col in features.columns:
                self._numeric_means[col] = float(features[col].mean())
                std = float(features[col].std())
                self._numeric_stds[col] = std if std > 0 else 1.0
                self._imputation_values[col] = float(features[col].median())

        # 2. Learn categorical encoding values
        for col in self.CATEGORICAL_FEATURES:
            if col in features.columns:
                self._category_values[col] = sorted(
                    features[col].dropna().unique().tolist()
                )

        # 3. Learn imputation values for binary
        for col in self.BINARY_FEATURES:
            if col in features.columns:
                self._imputation_values[col] = int(features[col].mode().iloc[0])

        self._is_fitted = True
        logger.info(f"Pipeline fitted. Numeric: {len(self._numeric_means)}, "
                     f"Categorical: {len(self._category_values)}")

        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw features into model-ready format.

        Applies:
        - Missing value imputation
        - Numeric standardization (zero mean, unit variance)
        - Cyclical encoding for month and day_of_week
        - One-hot encoding for categorical features
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        result = pd.DataFrame(index=features.index)

        # 1. Numeric features — impute + standardize
        for col in self.NUMERIC_FEATURES:
            if col in features.columns:
                values = features[col].fillna(
                    self._imputation_values.get(col, 0)
                ).astype(float)

                mean = self._numeric_means.get(col, 0)
                std = self._numeric_stds.get(col, 1)
                result[col] = (values - mean) / std
            else:
                result[col] = 0.0

        # 2. Binary features — impute only
        for col in self.BINARY_FEATURES:
            if col in features.columns:
                result[col] = features[col].fillna(
                    self._imputation_values.get(col, 0)
                ).astype(int)
            else:
                result[col] = 0

        # 3. Cyclical features — sin/cos encoding
        for col in self.CYCLICAL_FEATURES:
            if col in features.columns:
                if col == 'month':
                    period = 12
                elif col == 'day_of_week':
                    period = 7
                else:
                    period = 12

                values = features[col].fillna(0).astype(float)
                result[f'{col}_sin'] = np.sin(2 * np.pi * values / period)
                result[f'{col}_cos'] = np.cos(2 * np.pi * values / period)

        # 4. Categorical features — one-hot encoding
        for col in self.CATEGORICAL_FEATURES:
            known_values = self._category_values.get(col, [])
            if col in features.columns:
                for val in known_values:
                    result[f'{col}_{val}'] = (features[col] == val).astype(int)
            else:
                for val in known_values:
                    result[f'{col}_{val}'] = 0

        self._feature_names = result.columns.tolist()
        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
        order_history: Optional[pd.DataFrame] = None,
        price_history: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Extract features, fit pipeline, and transform in one step."""
        features = self.extract_features(data, order_history, price_history)
        self.fit(features)
        return self.transform(features)

    def transform_single(
        self,
        menu_item_id: str,
        quantity: int,
        event_date: Optional[str] = None,
        guest_count: Optional[int] = None,
        event_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform a single prediction request into feature vector.
        Used during inference (API prediction).
        """
        data = pd.DataFrame([{
            'menu_item_id': menu_item_id,
            'quantity': quantity,
            'event_date': event_date,
            'guest_count': guest_count or 50,
            'event_type': event_type or 'Other',
        }])

        features = self.extract_features(data)
        return self.transform(features)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ── SERIALIZATION ──────────────────────────────────────

    def save(self, filepath: str):
        """Save fitted pipeline to disk."""
        state = {
            'numeric_means': self._numeric_means,
            'numeric_stds': self._numeric_stds,
            'category_values': self._category_values,
            'imputation_values': self._imputation_values,
            'is_fitted': self._is_fitted,
            'feature_names': self._feature_names,
            'version': '1.0',
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FeaturePipeline':
        """Load a fitted pipeline from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        pipeline = cls()
        pipeline._numeric_means = state['numeric_means']
        pipeline._numeric_stds = state['numeric_stds']
        pipeline._category_values = state['category_values']
        pipeline._imputation_values = state['imputation_values']
        pipeline._is_fitted = state['is_fitted']
        pipeline._feature_names = state['feature_names']

        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline

    def get_params(self) -> dict:
        """Get pipeline parameters for documentation/debugging."""
        return {
            'numeric_means': self._numeric_means,
            'numeric_stds': self._numeric_stds,
            'category_values': self._category_values,
            'imputation_values': self._imputation_values,
            'n_features': len(self._feature_names),
            'feature_names': self._feature_names,
        }


# ============================================================
# FEATURE ANALYSIS UTILITIES
# ============================================================

class FeatureAnalyzer:
    """
    Utilities for feature validation and selection.
    Used during model development (not in production).
    """

    @staticmethod
    def calculate_correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pearson correlation matrix for numeric features."""
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        return features[numeric_cols].corr()

    @staticmethod
    def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.9) -> List[dict]:
        """Find feature pairs with correlation above threshold."""
        high_corr = []
        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr.append({
                        'feature_1': cols[i],
                        'feature_2': cols[j],
                        'correlation': round(corr_val, 4),
                        'action': f'Consider removing one (keeping the more interpretable)'
                    })

        return high_corr

    @staticmethod
    def calculate_vif(features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor for each numeric feature.
        VIF > 10 indicates problematic multicollinearity.
        """
        from numpy.linalg import LinAlgError

        numeric_cols = features.select_dtypes(include=[np.number]).columns
        X = features[numeric_cols].dropna()

        if len(X) < 2 or len(numeric_cols) < 2:
            return pd.DataFrame({'feature': numeric_cols, 'VIF': [1.0] * len(numeric_cols)})

        vif_data = []
        for i, col in enumerate(numeric_cols):
            try:
                y = X[col].values
                X_others = X.drop(columns=[col]).values

                if X_others.shape[1] == 0:
                    vif_data.append({'feature': col, 'VIF': 1.0})
                    continue

                # Add constant
                X_with_const = np.column_stack([np.ones(len(X_others)), X_others])

                # OLS: y = X_others * beta
                try:
                    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    y_pred = X_with_const @ beta
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                except LinAlgError:
                    vif = float('inf')

                vif_data.append({'feature': col, 'VIF': round(vif, 2)})

            except Exception as e:
                vif_data.append({'feature': col, 'VIF': -1.0})

        return pd.DataFrame(vif_data)

    @staticmethod
    def extract_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from a trained model.
        Works with tree-based models (RandomForest, GradientBoosting, XGBoost).
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            raise ValueError("Model does not have feature_importances_ or coef_")

        df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances,
            'importance_pct': (importances / importances.sum() * 100)
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        df['cumulative_pct'] = df['importance_pct'].cumsum()
        df['rank'] = range(1, len(df) + 1)

        return df

    @staticmethod
    def validate_top_features(importance_df: pd.DataFrame) -> List[dict]:
        """
        Validate that top features make business sense.
        Flags potential data leakage if unexpected features dominate.
        """
        expected_top = {'base_ingredient_cost', 'quantity', 'recipe_complexity'}
        suspicious = {'menu_item_id', 'item_id', 'recipe_id', 'prediction_id'}

        validations = []
        top_5 = importance_df.head(5)['feature'].tolist()

        # Check expected features are in top 5
        for feat in expected_top:
            matching = [f for f in top_5 if feat in f]
            if matching:
                validations.append({
                    'check': f'Expected feature "{feat}" in top 5',
                    'status': 'PASS',
                    'details': f'Found at rank {top_5.index(matching[0]) + 1}'
                })
            else:
                validations.append({
                    'check': f'Expected feature "{feat}" in top 5',
                    'status': 'WARNING',
                    'details': 'Not in top 5 — verify model is learning correctly'
                })

        # Check for suspicious features
        for feat in top_5:
            for susp in suspicious:
                if susp in feat.lower():
                    validations.append({
                        'check': f'Suspicious feature "{feat}" in top 5',
                        'status': 'FAIL',
                        'details': 'Possible data leakage — ID features should not be important'
                    })

        return validations
