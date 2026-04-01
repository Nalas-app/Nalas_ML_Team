import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = pd.DataFrame({
    "event_type": np.random.choice(["Wedding", "Birthday", "Corporate"], n),
    "location": np.random.choice(["Coimbatore", "Chennai", "Bangalore"], n),
    "quantity": np.random.randint(50, 1000, n),
    "ingredient_cost": np.random.randint(10000, 200000, n),
    "labor_cost": np.random.randint(5000, 50000, n),
    "overhead_cost": np.random.randint(2000, 30000, n),
    "demand_index": np.random.uniform(0.8, 1.5, n)
})

# True cost formula (assumption formula)
data["total_cost"] = (
    (data["ingredient_cost"] + data["labor_cost"] + data["overhead_cost"])
    * data["demand_index"]
    + (0.15 * data["quantity"])
)

data.to_csv("dummy_catering_data.csv", index=False)

print("Dummy dataset created!")
