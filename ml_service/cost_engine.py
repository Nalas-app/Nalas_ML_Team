"""
Rule-Based Cost Calculation Engine
====================================
Phase 3 — Deterministic cost prediction using recipe data + business rules.

Algorithm:
    1. Input: menu_item_id, quantity
    2. Fetch recipe for menu item
    3. For each ingredient in recipe:
       - Get current ingredient cost
       - Calculate: ingredient_needed = recipe.quantity * order.quantity / base_batch_size
       - Calculate: ingredient_cost = ingredient_needed * current_price * wastage_factor
    4. Sum all ingredient costs -> total_ingredient_cost
    5. Apply business rules:
       - labor_cost = total_ingredient_cost * 0.15
       - overhead_cost = total_ingredient_cost * 0.10
       - wastage_cost = total_ingredient_cost * 0.05
       - subtotal = total_ingredient_cost + labor + overhead + wastage
       - profit_margin = subtotal * 0.10
       - final_cost = subtotal + profit_margin
    6. Return cost breakdown
"""

import json
import os
import logging
from datetime import datetime, date
from typing import Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger("ml_costing.engine")

# ============================================================
# DATA LOADER — Loads structured CSVs into memory
# ============================================================

class DataStore:
    """Loads and caches structured datasets from CSV files."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ingredients = None
        self.menu_items = None
        self.recipes = None
        self.recipe_costs = None
        self._loaded = False
        self._load_time = None

    def load(self):
        """Load all structured datasets into memory."""
        try:
            self.ingredients = pd.read_csv(
                os.path.join(self.data_dir, "ingredients_master.csv")
            )
            self.menu_items = pd.read_csv(
                os.path.join(self.data_dir, "menu_items.csv")
            )
            self.recipes = pd.read_csv(
                os.path.join(self.data_dir, "recipes.csv")
            )

            costs_path = os.path.join(self.data_dir, "recipe_costs.csv")
            if os.path.exists(costs_path):
                self.recipe_costs = pd.read_csv(costs_path)

            self._loaded = True
            self._load_time = datetime.now()

            logger.info(
                f"DataStore loaded: {len(self.menu_items)} items, "
                f"{len(self.ingredients)} ingredients, {len(self.recipes)} recipe rows"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self._loaded = False
            return False

    @property
    def is_loaded(self):
        return self._loaded

    def get_menu_item(self, item_id: str) -> Optional[dict]:
        """Look up menu item by ID or name."""
        if not self._loaded:
            return None

        # Try exact ID match
        match = self.menu_items[self.menu_items['item_id'] == item_id]
        if len(match) > 0:
            return match.iloc[0].to_dict()

        # Try name match (case-insensitive)
        match = self.menu_items[
            self.menu_items['item_name'].str.lower().str.strip() == item_id.lower().strip()
        ]
        if len(match) > 0:
            return match.iloc[0].to_dict()

        return None

    def get_recipe(self, item_id: str) -> pd.DataFrame:
        """Get recipe rows for a menu item."""
        if not self._loaded:
            return pd.DataFrame()
        return self.recipes[self.recipes['menu_item_id'] == item_id].copy()

    def get_ingredient_price(self, ingredient_id: str) -> Optional[float]:
        """Look up ingredient price by ID."""
        if not self._loaded or ingredient_id == 'UNMATCHED':
            return None
        match = self.ingredients[self.ingredients['ingredient_id'] == ingredient_id]
        if len(match) > 0:
            price = match.iloc[0]['price_per_unit']
            return float(price) if pd.notna(price) and price > 0 else None
        return None

    def get_ingredient_name(self, ingredient_id: str) -> str:
        """Get ingredient name by ID."""
        if not self._loaded:
            return "Unknown"
        match = self.ingredients[self.ingredients['ingredient_id'] == ingredient_id]
        if len(match) > 0:
            return str(match.iloc[0]['ingredient_name'])
        return "Unknown"


# ============================================================
# BUSINESS RULES — Loads config from JSON
# ============================================================

class BusinessRules:
    """Business rules and multipliers loaded from config."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.labor_pct = self.config['labor_pct']
        self.overhead_pct = self.config['overhead_pct']
        self.wastage_pct = self.config['wastage_pct']
        self.profit_pct = self.config['profit_pct']
        self.model_version = self.config['model_version']
        self.bulk_discounts = self.config['bulk_discounts']
        self.seasonal_adjustments = self.config['seasonal_adjustments']
        self.min_cost_per_serving = self.config['min_cost_per_serving']

    def get_seasonal_adjustment(self, event_date: Optional[str]) -> float:
        """Calculate seasonal price adjustment factor."""
        if event_date is None:
            return 0.0

        try:
            dt = date.fromisoformat(event_date)
            month = dt.month
        except (ValueError, TypeError):
            return 0.0

        for season_name, season_config in self.seasonal_adjustments.items():
            if month in season_config['months']:
                return season_config['adjustment']

        return 0.0

    def get_bulk_discount(self, quantity: int) -> float:
        """Calculate bulk discount percentage based on quantity."""
        for tier in self.bulk_discounts:
            if tier['min_qty'] <= quantity <= tier['max_qty']:
                return tier['discount']
        return 0.0

    def get_min_cost(self, category: str) -> float:
        """Get minimum cost per serving for a category."""
        return self.min_cost_per_serving.get(
            category,
            self.min_cost_per_serving.get('default', 40)
        )


# ============================================================
# COST ENGINE — The core calculation logic
# ============================================================

class CostEngine:
    """
    Rule-based cost calculation engine.

    Given a menu_item_id and quantity, calculates the full cost breakdown
    using recipe data, ingredient prices, and business rules.
    """

    def __init__(self, data_store: DataStore, business_rules: BusinessRules):
        self.data = data_store
        self.rules = business_rules
        self._prediction_count = 0
        self._error_count = 0

    def predict(
        self,
        menu_item_id: str,
        quantity: int,
        event_date: Optional[str] = None,
        guest_count: Optional[int] = None
    ) -> dict:
        """
        Calculate cost prediction for a menu item.

        Args:
            menu_item_id: ID or name of the menu item
            quantity: Number of servings requested
            event_date: Optional event date (YYYY-MM-DD) for seasonal adjustment
            guest_count: Optional guest count (for future use)

        Returns:
            dict with cost breakdown and metadata

        Raises:
            CostingError: If prediction cannot be completed
        """
        self._prediction_count += 1
        flags = []

        # ── STEP 1: Validate inputs ──
        if quantity <= 0:
            raise CostingError("ML_001", "Invalid quantity", {"quantity": quantity})

        # ── STEP 2: Look up menu item ──
        item = self.data.get_menu_item(menu_item_id)
        if item is None:
            raise CostingError(
                "ML_002",
                f"Menu item not found: {menu_item_id}",
                {"menu_item_id": menu_item_id}
            )

        actual_item_id = item['item_id']
        item_name = item['item_name']
        category = item.get('category', 'default')

        # ── STEP 3: Fetch recipe ──
        recipe = self.data.get_recipe(actual_item_id)
        if len(recipe) == 0:
            raise CostingError(
                "ML_003",
                f"Recipe not found for item: {item_name}",
                {"menu_item_id": actual_item_id, "item_name": item_name}
            )

        # ── STEP 4: Calculate ingredient costs ──
        total_ingredient_cost = 0.0
        matched_count = 0
        unmatched_count = 0
        missing_price_count = 0
        ingredient_breakdown = []

        for _, row in recipe.iterrows():
            ing_id = row['ingredient_id']
            ing_name = row.get('ingredient_name', 'Unknown')
            qty_per_base = row.get('quantity_per_base_unit', None)
            wastage_factor = row.get('wastage_factor', 1.05)

            # Skip if no quantity data
            if pd.isna(qty_per_base) or qty_per_base is None:
                unmatched_count += 1
                continue

            # Skip if ingredient not matched
            if ing_id == 'UNMATCHED':
                unmatched_count += 1
                continue

            # Get price
            price = self.data.get_ingredient_price(ing_id)
            if price is None or price <= 0:
                missing_price_count += 1
                flags.append(f"missing_price:{ing_name}")
                continue

            # Calculate cost for this ingredient
            # qty_per_base is the amount needed for 1 base batch
            # We scale by the ordered quantity (treating it as servings)
            ingredient_cost = qty_per_base * price * wastage_factor
            total_ingredient_cost += ingredient_cost
            matched_count += 1

            ingredient_breakdown.append({
                'ingredient': ing_name,
                'quantity': float(qty_per_base),
                'price_per_unit': float(price),
                'cost': round(float(ingredient_cost), 2)
            })

        # ── STEP 5: Check if we have enough data ──
        total_ingredients = matched_count + unmatched_count + missing_price_count
        if matched_count == 0:
            raise CostingError(
                "ML_004",
                f"Cannot calculate cost: no priced ingredients for {item_name}",
                {
                    "menu_item_id": actual_item_id,
                    "item_name": item_name,
                    "total_ingredients": total_ingredients,
                    "unmatched": unmatched_count,
                    "missing_prices": missing_price_count
                }
            )

        # Scale ingredient cost by quantity
        # The recipe is for 1 base batch, multiply by quantity ordered
        total_ingredient_cost = total_ingredient_cost * quantity

        # ── STEP 6: Apply business multipliers ──
        labor_cost = total_ingredient_cost * self.rules.labor_pct
        overhead_cost = total_ingredient_cost * self.rules.overhead_pct
        wastage_cost = total_ingredient_cost * self.rules.wastage_pct

        subtotal = total_ingredient_cost + labor_cost + overhead_cost + wastage_cost

        # Apply seasonal adjustment (adds to subtotal)
        seasonal_adj = self.rules.get_seasonal_adjustment(event_date)
        if seasonal_adj > 0:
            seasonal_amount = subtotal * seasonal_adj
            subtotal += seasonal_amount
            flags.append(f"seasonal_adjustment:{seasonal_adj:.0%}")

        # Apply profit margin
        profit_margin = subtotal * self.rules.profit_pct
        final_cost = subtotal + profit_margin

        # Apply bulk discount
        bulk_discount = self.rules.get_bulk_discount(quantity)
        if bulk_discount > 0:
            discount_amount = final_cost * bulk_discount
            final_cost -= discount_amount
            flags.append(f"bulk_discount:{bulk_discount:.0%}")

        # ── STEP 7: Enforce minimum cost ──
        min_cost = self.rules.get_min_cost(category) * quantity
        if final_cost < min_cost:
            flags.append(f"min_cost_enforced: {min_cost}")
            final_cost = min_cost

        # ── STEP 8: Ensure all costs are positive ──
        final_cost = max(final_cost, 0.01)

        # ── STEP 9: Calculate confidence ──
        completeness = matched_count / max(total_ingredients, 1)
        confidence = self._calculate_confidence(completeness, matched_count, unmatched_count)

        if confidence < 0.60:
            flags.append("low_confidence")

        # ── STEP 10: Build response ──
        result = {
            "ingredientCost": round(total_ingredient_cost, 2),
            "laborCost": round(labor_cost, 2),
            "overheadCost": round(overhead_cost, 2),
            "totalCost": round(final_cost, 2),
            "confidence": round(confidence, 2),
            "modelVersion": self.rules.model_version,
            "method": "rule_based",
            # Extra metadata (for logging, not in API response)
            "_meta": {
                "item_id": actual_item_id,
                "item_name": item_name,
                "category": category,
                "quantity": quantity,
                "ingredients_matched": matched_count,
                "ingredients_unmatched": unmatched_count,
                "ingredients_missing_price": missing_price_count,
                "seasonal_adjustment": seasonal_adj,
                "bulk_discount": bulk_discount,
                "flags": flags,
                "ingredient_breakdown": ingredient_breakdown
            }
        }

        logger.info(
            f"Prediction: {item_name} x{quantity} = Rs.{final_cost:.2f} "
            f"(confidence: {confidence:.0%}, matched: {matched_count}/{total_ingredients})"
        )

        return result

    def _calculate_confidence(
        self,
        completeness: float,
        matched: int,
        unmatched: int
    ) -> float:
        """
        Calculate confidence score based on data quality.

        Factors:
        - Ingredient match rate (primary)
        - Number of matched ingredients (more = higher confidence)
        - Method is rule-based (capped at 0.85 — ML can go higher)
        """
        # Base confidence from completeness
        base = completeness

        # Bonus for having more matched ingredients (diminishing returns)
        if matched >= 10:
            bonus = 0.10
        elif matched >= 5:
            bonus = 0.05
        else:
            bonus = 0.0

        confidence = min(base + bonus, 0.85)  # Rule-based cap: 85%

        # Penalty if many ingredients are unmatched
        if unmatched > matched:
            confidence *= 0.7

        return max(confidence, 0.10)  # Floor: 10%

    @property
    def prediction_count(self):
        return self._prediction_count

    @property
    def error_count(self):
        return self._error_count


# ============================================================
# CUSTOM ERROR
# ============================================================

class CostingError(Exception):
    """Custom exception for costing errors with error codes."""

    def __init__(self, code: str, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
