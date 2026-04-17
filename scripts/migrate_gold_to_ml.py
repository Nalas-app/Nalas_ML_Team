import pandas as pd
import os

# Paths
GOLD_DIR = r"c:\Users\jaisu\Projects\nalas\nrc-datasets\gold"
ML_DATA_DIR = r"c:\Users\jaisu\Projects\nalas\Nalas_ML_Team_Clone\data\structured"

def migrate():
    print("--- SYNCING GOLD DATA TO ML SERVICE ---")
    
    # 1. Ingredients
    print("Migrating Ingredients...")
    gold_ing = pd.read_csv(os.path.join(GOLD_DIR, "gold_ingredients.csv"))
    # ML expects: ingredient_id,ingredient_name,category,unit,price_per_unit,is_perishable,last_updated
    ml_ing = pd.DataFrame()
    ml_ing['ingredient_id'] = gold_ing['ingredient_id']
    ml_ing['ingredient_name'] = gold_ing['name']
    ml_ing['category'] = gold_ing['category']
    ml_ing['unit'] = gold_ing['unit']
    ml_ing['price_per_unit'] = gold_ing['price_per_unit']
    ml_ing['is_perishable'] = False
    ml_ing['last_updated'] = "2026-04-17"
    ml_ing.to_csv(os.path.join(ML_DATA_DIR, "ingredients_master.csv"), index=False)
    
    # 2. Menu Items
    print("Migrating Menu Items...")
    gold_menu = pd.read_csv(os.path.join(GOLD_DIR, "gold_menu_items.csv"))
    # ML expects: item_id,item_name,category,base_unit,source
    ml_menu = pd.DataFrame()
    ml_menu['item_id'] = gold_menu['menu_item_id']
    ml_menu['item_name'] = gold_menu['name']
    ml_menu['category'] = gold_menu['category']
    ml_menu['base_unit'] = gold_menu['base_unit']
    ml_menu['source'] = "nrc_gold"
    ml_menu.to_csv(os.path.join(ML_DATA_DIR, "menu_items.csv"), index=False)
    
    # 3. Recipes
    print("Migrating Recipes...")
    gold_rcp = pd.read_csv(os.path.join(GOLD_DIR, "gold_recipes.csv"))
    # ML expects: recipe_id,menu_item_id,ingredient_id,quantity_per_base_unit,wastage_factor
    ml_rcp = pd.DataFrame()
    ml_rcp['recipe_id'] = gold_rcp['recipe_id']
    ml_rcp['menu_item_id'] = gold_rcp['menu_item_id']
    ml_rcp['ingredient_id'] = gold_rcp['ingredient_id']
    ml_rcp['quantity_per_base_unit'] = gold_rcp['quantity_per_base_unit']
    ml_rcp['wastage_factor'] = gold_rcp['wastage_factor']
    ml_rcp.to_csv(os.path.join(ML_DATA_DIR, "recipes.csv"), index=False)

    print("✅ Sync Complete!")

if __name__ == "__main__":
    migrate()
