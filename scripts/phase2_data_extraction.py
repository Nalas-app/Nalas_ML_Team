"""
Phase 2.2 — Data Extraction & Structuring Script
=================================================
Project: Nalas ML Costing Engine
Author: Adwaitha (ML Costing Lead)
Date: 2026-03-28

This script:
1. Parses recipe data from NRC Excel files
2. Creates structured datasets (menu_items, recipes, ingredients, recipe_costs)
3. Validates data integrity
4. Exports to CSV for analysis
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import uuid
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

MENU_FILE = os.path.join(PROJECT_ROOT, "NRC  MENU.xlsx")
KITCHEN_FILE = os.path.join(PROJECT_ROOT, "NRC Kithen_Outsource-Inn.xls")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "structured")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_DIR = os.path.join(BASE_DIR, "docs", "phase2")
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_id():
    """Generate a short deterministic-style ID."""
    return str(uuid.uuid4())[:8]


# ============================================================
# STEP 1: EXTRACT INGREDIENT MASTER (Prices)
# ============================================================
def extract_ingredients():
    """Extract ingredient master with prices from Kitchen file."""
    print("\n" + "=" * 60)
    print("STEP 1: Extracting Ingredient Master")
    print("=" * 60)

    prices = pd.read_excel(KITCHEN_FILE, sheet_name='Prices')

    # Identify actual column names
    print(f"\nRaw columns: {list(prices.columns)}")
    print(f"Raw rows: {len(prices)}")

    # Try to extract relevant columns
    # The Kitchen file has: Item Name, Category Name, Purchasing Measurement Unit, Rate
    possible_name_cols = [c for c in prices.columns if 'item' in c.lower() or 'name' in c.lower()]
    possible_cat_cols = [c for c in prices.columns if 'categ' in c.lower()]
    possible_unit_cols = [c for c in prices.columns if 'unit' in c.lower() or 'measur' in c.lower()]
    possible_rate_cols = [c for c in prices.columns if 'rate' in c.lower() or 'price' in c.lower() or 'cost' in c.lower()]

    print(f"\nDetected columns:")
    print(f"  Name cols: {possible_name_cols}")
    print(f"  Category cols: {possible_cat_cols}")
    print(f"  Unit cols: {possible_unit_cols}")
    print(f"  Rate cols: {possible_rate_cols}")

    # Build ingredients dataframe
    try:
        ingredients_df = prices[['Item Name', 'Category Name', 'Purchasing Measurement Unit', 'Rate']].copy()
        ingredients_df.columns = ['ingredient_name', 'category', 'unit', 'price_per_unit']
    except KeyError:
        # Fallback: use positional
        print("  [WARN] Column names don't match expected. Using available columns.")
        cols = prices.columns.tolist()
        ingredients_df = prices.iloc[:, :4].copy()
        ingredients_df.columns = ['ingredient_name', 'category', 'unit', 'price_per_unit']

    # Clean data
    ingredients_df = ingredients_df.dropna(subset=['ingredient_name'])
    ingredients_df['ingredient_name'] = ingredients_df['ingredient_name'].astype(str).str.strip()
    ingredients_df['category'] = ingredients_df['category'].astype(str).str.strip()
    ingredients_df['unit'] = ingredients_df['unit'].astype(str).str.strip()

    # Convert price to numeric
    ingredients_df['price_per_unit'] = pd.to_numeric(ingredients_df['price_per_unit'], errors='coerce')

    # Add ingredient IDs
    ingredients_df.insert(0, 'ingredient_id', [f"ING-{str(i+1).zfill(3)}" for i in range(len(ingredients_df))])

    # Add metadata
    ingredients_df['is_perishable'] = ingredients_df['category'].apply(
        lambda x: True if x.lower() in ['vegetables', 'dairy', 'meat', 'fish', 'fruits']
        else False
    )
    ingredients_df['last_updated'] = datetime.now().strftime('%Y-%m-%d')

    print(f"\n✅ Extracted {len(ingredients_df)} ingredients")
    print(f"   With prices: {ingredients_df['price_per_unit'].notna().sum()}")
    print(f"   Missing prices: {ingredients_df['price_per_unit'].isna().sum()}")
    print(f"   Categories: {ingredients_df['category'].nunique()}")

    # Category distribution
    print(f"\n--- Category Distribution ---")
    print(ingredients_df['category'].value_counts().to_string())

    return ingredients_df


# ============================================================
# STEP 2: EXTRACT RECIPE DATA
# ============================================================
def extract_recipes(ingredients_df):
    """Extract recipe data from Kitchen file (sheets with cost breakdowns)."""
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Recipe Data")
    print("=" * 60)

    # Known recipe sheets with complete data from Kitchen file
    recipe_sheets = {
        'KCF': {'name': 'Chicken Fry (KCF)', 'category': 'Non-Veg Main Course', 'base_unit': '6kg batch'},
        'C Curry': {'name': 'Chicken Curry', 'category': 'Non-Veg Main Course', 'base_unit': '1 batch'},
        'Changezi': {'name': 'Chicken Changezi', 'category': 'Non-Veg Main Course', 'base_unit': '1 batch'},
        'Ghee Rice': {'name': 'Ghee Rice', 'category': 'Rice', 'base_unit': '2kg batch'},
        'M Varuval': {'name': 'Mutton Varuval', 'category': 'Non-Veg Main Course', 'base_unit': '1 batch'},
        'Pongal': {'name': 'Pongal', 'category': 'Breakfast', 'base_unit': '1 batch'},
        'Sambar': {'name': 'Sambar', 'category': 'Side Dish', 'base_unit': '1 batch'},
    }

    # Also get recipe sheets from Menu file
    menu_recipe_sheets = [
        'Lunch Sambar', 'Rasam', 'Pasi Paruppu', 'Kootu', 'Poriyal',
        'Aviyal', 'Thogayal', 'Pickle', 'Payasam', 'Curd Rice',
        'Lemon Rice', 'Tomato Rice', 'Puliyogare', 'Bisibela Bath',
        'Veg Biryani', 'Paneer Butter Masala', 'Gobi Manchurian',
        'Mushroom Masala', 'Chapati', 'Poori', 'Parotta', 'Idli',
        'Dosa', 'Vada'
    ]

    menu_items = []
    recipes = []
    recipe_id_counter = 1

    # Extract from Kitchen file
    for sheet_name, info in recipe_sheets.items():
        try:
            sheet_data = pd.read_excel(KITCHEN_FILE, sheet_name=sheet_name)
            item_id = f"ITEM-{str(recipe_id_counter).zfill(3)}"

            menu_items.append({
                'item_id': item_id,
                'item_name': info['name'],
                'category': info['category'],
                'base_unit': info['base_unit'],
                'description': f"Recipe from NRC Kitchen file - {sheet_name}",
                'source': 'Kitchen File',
                'has_quantities': True,
                'has_cost_data': True
            })

            # Try to extract ingredient rows from sheet
            for _, row in sheet_data.iterrows():
                row_values = row.dropna().values
                if len(row_values) >= 2:
                    ingredient_name = str(row_values[0]).strip()
                    # Try to find quantity
                    qty = None
                    for val in row_values[1:]:
                        try:
                            qty = float(val)
                            break
                        except (ValueError, TypeError):
                            continue

                    if qty is not None and ingredient_name and len(ingredient_name) > 1:
                        # Match to ingredient master
                        matched = ingredients_df[
                            ingredients_df['ingredient_name'].str.lower().str.contains(
                                ingredient_name.lower()[:5], na=False
                            )
                        ]
                        ing_id = matched.iloc[0]['ingredient_id'] if len(matched) > 0 else 'UNMATCHED'

                        recipes.append({
                            'recipe_id': f"RCP-{str(len(recipes)+1).zfill(4)}",
                            'menu_item_id': item_id,
                            'ingredient_name': ingredient_name,
                            'ingredient_id': ing_id,
                            'quantity_per_base_unit': qty,
                            'wastage_factor': 1.05  # Default 5% wastage
                        })

            recipe_id_counter += 1
            print(f"  ✅ {sheet_name}: {info['name']} — extracted")

        except Exception as e:
            print(f"  ⚠️ {sheet_name}: Could not parse — {e}")

    # Extract from Menu file (most have NO quantities)
    try:
        menu_xl = pd.ExcelFile(MENU_FILE)
        menu_sheet_names = menu_xl.sheet_names
        print(f"\n--- NRC MENU.xlsx sheets: {len(menu_sheet_names)} ---")

        skip_sheets = ['Inventory List', 'Sheet2', 'Price']
        for sheet_name in menu_sheet_names:
            if sheet_name in skip_sheets:
                continue

            item_id = f"ITEM-{str(recipe_id_counter).zfill(3)}"
            try:
                sheet_data = pd.read_excel(MENU_FILE, sheet_name=sheet_name)
                has_qty = sheet_data.shape[0] > 0 and sheet_data.shape[1] >= 2

                # Check if columns have actual numeric data
                numeric_cols = sheet_data.select_dtypes(include=[np.number]).columns
                has_numeric = len(numeric_cols) > 0 and sheet_data[numeric_cols].notna().sum().sum() > 0

                menu_items.append({
                    'item_id': item_id,
                    'item_name': sheet_name,
                    'category': 'Veg Main Course',  # Default, needs manual mapping
                    'base_unit': '25 person batch',
                    'description': f"Recipe from NRC Menu file - {sheet_name}",
                    'source': 'Menu File',
                    'has_quantities': has_numeric,
                    'has_cost_data': False
                })

                # Try extracting ingredients
                for _, row in sheet_data.iterrows():
                    row_values = row.dropna().values
                    if len(row_values) >= 1:
                        ingredient_name = str(row_values[0]).strip()
                        qty = None
                        if len(row_values) >= 2:
                            try:
                                qty = float(row_values[1])
                            except (ValueError, TypeError):
                                qty = None

                        if ingredient_name and len(ingredient_name) > 1:
                            matched = ingredients_df[
                                ingredients_df['ingredient_name'].str.lower().str.contains(
                                    ingredient_name.lower()[:5], na=False
                                )
                            ]
                            ing_id = matched.iloc[0]['ingredient_id'] if len(matched) > 0 else 'UNMATCHED'

                            recipes.append({
                                'recipe_id': f"RCP-{str(len(recipes)+1).zfill(4)}",
                                'menu_item_id': item_id,
                                'ingredient_name': ingredient_name,
                                'ingredient_id': ing_id,
                                'quantity_per_base_unit': qty,
                                'wastage_factor': 1.05
                            })

                recipe_id_counter += 1
                status = "✅" if has_numeric else "⚠️ NO QTY"
                print(f"  {status} {sheet_name}")

            except Exception as e:
                print(f"  ❌ {sheet_name}: Error — {e}")
                recipe_id_counter += 1

    except Exception as e:
        print(f"  ❌ Could not read Menu file: {e}")

    menu_items_df = pd.DataFrame(menu_items)
    recipes_df = pd.DataFrame(recipes)

    print(f"\n✅ Total menu items: {len(menu_items_df)}")
    print(f"   With quantities: {menu_items_df['has_quantities'].sum()}")
    print(f"   Without quantities: {(~menu_items_df['has_quantities']).sum()}")
    print(f"✅ Total recipe rows: {len(recipes_df)}")
    print(f"   With quantities: {recipes_df['quantity_per_base_unit'].notna().sum()}")
    print(f"   Matched ingredients: {(recipes_df['ingredient_id'] != 'UNMATCHED').sum()}")
    print(f"   Unmatched ingredients: {(recipes_df['ingredient_id'] == 'UNMATCHED').sum()}")

    return menu_items_df, recipes_df


# ============================================================
# STEP 3: CALCULATE COSTS
# ============================================================
def calculate_costs(menu_items_df, recipes_df, ingredients_df):
    """Calculate cost breakdown for each menu item."""
    print("\n" + "=" * 60)
    print("STEP 3: Calculating Costs")
    print("=" * 60)

    # Business multipliers
    LABOR_PCT = 0.15
    OVERHEAD_PCT = 0.10
    WASTAGE_PCT = 0.05
    PROFIT_PCT = 0.10

    recipe_costs = []
    anomalies = []

    for _, item in menu_items_df.iterrows():
        item_id = item['item_id']
        item_name = item['item_name']

        # Get recipe ingredients for this item
        item_recipes = recipes_df[recipes_df['menu_item_id'] == item_id].copy()

        if len(item_recipes) == 0:
            anomalies.append({
                'item_id': item_id,
                'item_name': item_name,
                'issue': 'No recipe data found',
                'severity': 'HIGH'
            })
            continue

        # Calculate ingredient cost
        total_ingredient_cost = 0
        ingredients_with_cost = 0
        ingredients_without_cost = 0

        for _, recipe_row in item_recipes.iterrows():
            ing_id = recipe_row['ingredient_id']
            qty = recipe_row['quantity_per_base_unit']

            if ing_id == 'UNMATCHED' or pd.isna(qty):
                ingredients_without_cost += 1
                continue

            # Look up price
            ing_match = ingredients_df[ingredients_df['ingredient_id'] == ing_id]
            if len(ing_match) > 0 and pd.notna(ing_match.iloc[0]['price_per_unit']):
                price = ing_match.iloc[0]['price_per_unit']
                cost = qty * price * recipe_row.get('wastage_factor', 1.05)
                total_ingredient_cost += cost
                ingredients_with_cost += 1
            else:
                ingredients_without_cost += 1

        if total_ingredient_cost == 0:
            anomalies.append({
                'item_id': item_id,
                'item_name': item_name,
                'issue': 'Could not calculate ingredient cost (missing quantities or prices)',
                'severity': 'HIGH'
            })
            continue

        # Apply business multipliers
        labor_cost = total_ingredient_cost * LABOR_PCT
        overhead_cost = total_ingredient_cost * OVERHEAD_PCT
        wastage_cost = total_ingredient_cost * WASTAGE_PCT
        subtotal = total_ingredient_cost + labor_cost + overhead_cost + wastage_cost
        profit_margin = subtotal * PROFIT_PCT
        final_cost = subtotal + profit_margin

        recipe_costs.append({
            'menu_item_id': item_id,
            'item_name': item_name,
            'category': item['category'],
            'total_ingredient_cost': round(total_ingredient_cost, 2),
            'labor_cost': round(labor_cost, 2),
            'overhead_cost': round(overhead_cost, 2),
            'wastage_cost': round(wastage_cost, 2),
            'subtotal': round(subtotal, 2),
            'profit_margin': round(profit_margin, 2),
            'final_cost_per_unit': round(final_cost, 2),
            'ingredients_with_cost': ingredients_with_cost,
            'ingredients_without_cost': ingredients_without_cost,
            'cost_completeness_pct': round(
                ingredients_with_cost / max(ingredients_with_cost + ingredients_without_cost, 1) * 100, 1
            )
        })

        # Anomaly detection
        if ingredients_without_cost > 0:
            anomalies.append({
                'item_id': item_id,
                'item_name': item_name,
                'issue': f'{ingredients_without_cost} ingredients missing cost data',
                'severity': 'MEDIUM'
            })

    recipe_costs_df = pd.DataFrame(recipe_costs)
    anomalies_df = pd.DataFrame(anomalies)

    if len(recipe_costs_df) > 0:
        print(f"\n✅ Costs calculated for {len(recipe_costs_df)} menu items")
        print(f"\n--- Cost Summary ---")
        print(f"  Min final cost: ₹{recipe_costs_df['final_cost_per_unit'].min():,.2f}")
        print(f"  Max final cost: ₹{recipe_costs_df['final_cost_per_unit'].max():,.2f}")
        print(f"  Mean final cost: ₹{recipe_costs_df['final_cost_per_unit'].mean():,.2f}")
        print(f"  Median final cost: ₹{recipe_costs_df['final_cost_per_unit'].median():,.2f}")
    else:
        print("\n⚠️ No costs could be calculated — data gaps too large")

    print(f"\n⚠️ Anomalies found: {len(anomalies_df)}")
    if len(anomalies_df) > 0:
        print(anomalies_df[['item_name', 'issue', 'severity']].to_string(index=False))

    return recipe_costs_df, anomalies_df


# ============================================================
# STEP 4: DATA VALIDATION
# ============================================================
def validate_data(menu_items_df, recipes_df, ingredients_df, recipe_costs_df):
    """Run data quality checks."""
    print("\n" + "=" * 60)
    print("STEP 4: Data Validation")
    print("=" * 60)

    checks = []

    # Check 1: All menu items have recipes
    items_with_recipes = recipes_df['menu_item_id'].unique()
    items_without_recipes = menu_items_df[~menu_items_df['item_id'].isin(items_with_recipes)]
    checks.append({
        'check': 'All menu items have recipes',
        'status': '✅ PASS' if len(items_without_recipes) == 0 else '❌ FAIL',
        'details': f'{len(items_with_recipes)}/{len(menu_items_df)} items have recipe data',
        'missing': len(items_without_recipes)
    })

    # Check 2: All recipes have ingredient quantities
    recipes_with_qty = recipes_df['quantity_per_base_unit'].notna().sum()
    recipes_total = len(recipes_df)
    pct_with_qty = recipes_with_qty / max(recipes_total, 1) * 100
    checks.append({
        'check': 'All recipes have ingredient quantities',
        'status': '✅ PASS' if pct_with_qty > 90 else '⚠️ PARTIAL' if pct_with_qty > 50 else '❌ FAIL',
        'details': f'{recipes_with_qty}/{recipes_total} recipe rows have quantities ({pct_with_qty:.1f}%)',
        'missing': recipes_total - recipes_with_qty
    })

    # Check 3: All ingredients have costs
    ing_with_cost = ingredients_df['price_per_unit'].notna().sum()
    ing_total = len(ingredients_df)
    pct_with_cost = ing_with_cost / max(ing_total, 1) * 100
    checks.append({
        'check': 'All ingredients have costs',
        'status': '✅ PASS' if pct_with_cost > 95 else '⚠️ PARTIAL' if pct_with_cost > 80 else '❌ FAIL',
        'details': f'{ing_with_cost}/{ing_total} ingredients have prices ({pct_with_cost:.1f}%)',
        'missing': ing_total - ing_with_cost
    })

    # Check 4: No negative quantities
    neg_qty = (recipes_df['quantity_per_base_unit'] < 0).sum() if 'quantity_per_base_unit' in recipes_df else 0
    checks.append({
        'check': 'No negative quantities',
        'status': '✅ PASS' if neg_qty == 0 else '❌ FAIL',
        'details': f'{neg_qty} negative quantities found',
        'missing': neg_qty
    })

    # Check 5: No negative costs
    neg_costs = (ingredients_df['price_per_unit'] < 0).sum()
    checks.append({
        'check': 'No negative costs',
        'status': '✅ PASS' if neg_costs == 0 else '❌ FAIL',
        'details': f'{neg_costs} negative costs found',
        'missing': neg_costs
    })

    # Check 6: Units consistency
    unique_units = ingredients_df['unit'].nunique()
    checks.append({
        'check': 'Units are consistent',
        'status': '⚠️ REVIEW' if unique_units > 10 else '✅ PASS',
        'details': f'{unique_units} unique units found: {ingredients_df["unit"].unique().tolist()[:10]}',
        'missing': 0
    })

    # Check 7: Foreign key integrity
    unmatched_ings = (recipes_df['ingredient_id'] == 'UNMATCHED').sum()
    checks.append({
        'check': 'All recipe ingredients matched to master',
        'status': '✅ PASS' if unmatched_ings == 0 else '⚠️ PARTIAL',
        'details': f'{unmatched_ings} unmatched ingredient references',
        'missing': unmatched_ings
    })

    checks_df = pd.DataFrame(checks)
    print("\n--- Data Validation Results ---")
    for _, check in checks_df.iterrows():
        print(f"  {check['status']}  {check['check']}")
        print(f"           {check['details']}")

    return checks_df


# ============================================================
# STEP 5: GENERATE STATISTICS
# ============================================================
def generate_statistics(menu_items_df, recipes_df, ingredients_df, recipe_costs_df):
    """Generate comprehensive data statistics."""
    print("\n" + "=" * 60)
    print("STEP 5: Data Statistics")
    print("=" * 60)

    stats = {
        'generated_at': datetime.now().isoformat(),
        'menu_items': {
            'total': len(menu_items_df),
            'with_quantities': int(menu_items_df['has_quantities'].sum()),
            'without_quantities': int((~menu_items_df['has_quantities']).sum()),
            'by_category': menu_items_df['category'].value_counts().to_dict(),
            'by_source': menu_items_df['source'].value_counts().to_dict() if 'source' in menu_items_df else {}
        },
        'ingredients': {
            'total': len(ingredients_df),
            'with_prices': int(ingredients_df['price_per_unit'].notna().sum()),
            'without_prices': int(ingredients_df['price_per_unit'].isna().sum()),
            'by_category': ingredients_df['category'].value_counts().to_dict(),
            'price_stats': {
                'min': float(ingredients_df['price_per_unit'].min()) if ingredients_df['price_per_unit'].notna().any() else None,
                'max': float(ingredients_df['price_per_unit'].max()) if ingredients_df['price_per_unit'].notna().any() else None,
                'mean': float(ingredients_df['price_per_unit'].mean()) if ingredients_df['price_per_unit'].notna().any() else None,
                'median': float(ingredients_df['price_per_unit'].median()) if ingredients_df['price_per_unit'].notna().any() else None
            }
        },
        'recipes': {
            'total_rows': len(recipes_df),
            'unique_items_with_recipes': int(recipes_df['menu_item_id'].nunique()),
            'with_quantities': int(recipes_df['quantity_per_base_unit'].notna().sum()),
            'without_quantities': int(recipes_df['quantity_per_base_unit'].isna().sum()),
            'matched_ingredients': int((recipes_df['ingredient_id'] != 'UNMATCHED').sum()),
            'unmatched_ingredients': int((recipes_df['ingredient_id'] == 'UNMATCHED').sum()),
            'avg_ingredients_per_item': round(recipes_df.groupby('menu_item_id').size().mean(), 1) if len(recipes_df) > 0 else 0,
            'max_ingredients_per_item': int(recipes_df.groupby('menu_item_id').size().max()) if len(recipes_df) > 0 else 0,
            'min_ingredients_per_item': int(recipes_df.groupby('menu_item_id').size().min()) if len(recipes_df) > 0 else 0
        }
    }

    if len(recipe_costs_df) > 0:
        stats['costs'] = {
            'items_with_costs': len(recipe_costs_df),
            'cost_stats': {
                'min': float(recipe_costs_df['final_cost_per_unit'].min()),
                'max': float(recipe_costs_df['final_cost_per_unit'].max()),
                'mean': round(float(recipe_costs_df['final_cost_per_unit'].mean()), 2),
                'median': round(float(recipe_costs_df['final_cost_per_unit'].median()), 2),
                'std': round(float(recipe_costs_df['final_cost_per_unit'].std()), 2)
            },
            'ingredient_cost_stats': {
                'min': float(recipe_costs_df['total_ingredient_cost'].min()),
                'max': float(recipe_costs_df['total_ingredient_cost'].max()),
                'mean': round(float(recipe_costs_df['total_ingredient_cost'].mean()), 2)
            }
        }

    # Print summary
    print(f"\n📊 DATA STATISTICS SUMMARY")
    print(f"──────────────────────────")
    print(f"  Menu Items:      {stats['menu_items']['total']}")
    print(f"  Ingredients:     {stats['ingredients']['total']}")
    print(f"  Recipe Rows:     {stats['recipes']['total_rows']}")
    print(f"  Avg Ingredients/Item: {stats['recipes']['avg_ingredients_per_item']}")

    if 'costs' in stats:
        print(f"\n  💰 Cost Distribution:")
        print(f"     Min:    ₹{stats['costs']['cost_stats']['min']:,.2f}")
        print(f"     Max:    ₹{stats['costs']['cost_stats']['max']:,.2f}")
        print(f"     Mean:   ₹{stats['costs']['cost_stats']['mean']:,.2f}")
        print(f"     Median: ₹{stats['costs']['cost_stats']['median']:,.2f}")

    return stats


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("=" * 60)
    print("  NALAS ML COSTING — Phase 2 Data Extraction")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check files exist
    if not os.path.exists(KITCHEN_FILE):
        print(f"❌ Kitchen file not found: {KITCHEN_FILE}")
        print("   Please ensure NRC Kithen_Outsource-Inn.xls is in the project root")
        return
    if not os.path.exists(MENU_FILE):
        print(f"❌ Menu file not found: {MENU_FILE}")
        print("   Please ensure NRC  MENU.xlsx is in the project root")
        return

    print(f"✅ Kitchen file: {KITCHEN_FILE}")
    print(f"✅ Menu file: {MENU_FILE}")

    # Step 1: Extract ingredients
    ingredients_df = extract_ingredients()

    # Step 2: Extract recipes and menu items
    menu_items_df, recipes_df = extract_recipes(ingredients_df)

    # Step 3: Calculate costs
    recipe_costs_df, anomalies_df = calculate_costs(menu_items_df, recipes_df, ingredients_df)

    # Step 4: Validate data
    checks_df = validate_data(menu_items_df, recipes_df, ingredients_df, recipe_costs_df)

    # Step 5: Generate statistics
    stats = generate_statistics(menu_items_df, recipes_df, ingredients_df, recipe_costs_df)

    # =========================================================
    # EXPORT STRUCTURED DATASETS
    # =========================================================
    print("\n" + "=" * 60)
    print("EXPORTING DATASETS")
    print("=" * 60)

    # CSV exports
    ingredients_df.to_csv(os.path.join(OUTPUT_DIR, "ingredients_master.csv"), index=False)
    menu_items_df.to_csv(os.path.join(OUTPUT_DIR, "menu_items.csv"), index=False)
    recipes_df.to_csv(os.path.join(OUTPUT_DIR, "recipes.csv"), index=False)
    if len(recipe_costs_df) > 0:
        recipe_costs_df.to_csv(os.path.join(OUTPUT_DIR, "recipe_costs.csv"), index=False)
    if len(anomalies_df) > 0:
        anomalies_df.to_csv(os.path.join(OUTPUT_DIR, "cost_anomalies.csv"), index=False)
    checks_df.to_csv(os.path.join(OUTPUT_DIR, "validation_results.csv"), index=False)

    # JSON stats
    with open(os.path.join(OUTPUT_DIR, "data_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n✅ All datasets exported to: {OUTPUT_DIR}")
    print(f"   - ingredients_master.csv ({len(ingredients_df)} rows)")
    print(f"   - menu_items.csv ({len(menu_items_df)} rows)")
    print(f"   - recipes.csv ({len(recipes_df)} rows)")
    if len(recipe_costs_df) > 0:
        print(f"   - recipe_costs.csv ({len(recipe_costs_df)} rows)")
    if len(anomalies_df) > 0:
        print(f"   - cost_anomalies.csv ({len(anomalies_df)} rows)")
    print(f"   - validation_results.csv ({len(checks_df)} checks)")
    print(f"   - data_statistics.json")

    print("\n" + "=" * 60)
    print("  Phase 2 Data Extraction — COMPLETE")
    print("=" * 60)

    return ingredients_df, menu_items_df, recipes_df, recipe_costs_df, anomalies_df, checks_df, stats


if __name__ == "__main__":
    main()
