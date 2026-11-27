from pathlib import Path

import numpy as np
import pandas as pd

from battery import solve

folder = Path(__file__).resolve().parent
df = pd.read_csv(folder.parent / "training_data.csv")
prices = df["da_price_usd_per_mwh"].to_numpy(dtype=float)

# Load existing held-out predictions.
actions = pd.read_csv(folder / "holdout_terminal" / "actions.csv")
actions = actions.loc[actions["seed"] == 42].copy()

if actions.empty:
    raise ValueError("No predictions found for seed 42.")

# Match each prediction to its hourly row in training_data.csv.
actions["row"] = actions["day_index"] * 24 + actions["hour"] - 1
predicted = (
    actions.set_index("row")["predicted_price"]
    .reindex(np.arange(len(df)))
    .to_numpy()
)

# Whole-file perfect foresight, starting full once.
result = solve(prices, initial_energy=5.0, eta=1.0)

if not result["certified_optimal"]:
    raise RuntimeError("Solver did not certify optimality.")

output = pd.DataFrame({
    "price_usd_per_mwh": prices,
    "predicted_price_usd_per_mwh": predicted,
    "energy_end_mwh": result["energy"][1:],
    "change_mwh": result["schedule"].ravel(),
})

path = folder / "optimal_training_data.csv"
output.to_csv(path, index=False, float_format="%.6f")

print(f"Optimal total profit: ${result['revenue']:,.2f}")
print(f"Saved: {path}")