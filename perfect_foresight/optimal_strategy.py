from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import diags, eye, hstack

folder = Path(__file__).resolve().parent  # perfect_foresight/
data_folder = folder.parent              # energy-storage-project/


def find_optimal_strategy(filename):
    df = pd.read_csv(data_folder / filename)
    prices = df["da_price_usd_per_mwh"].to_numpy(dtype=float)
    n = len(prices)

    # Variables: n hourly actions followed by n end-of-hour energies.
    # Positive action = buy; negative action = sell.
    objective = np.concatenate([prices, np.zeros(n)])

    # Energy dynamics:
    # energy[t] - energy[t-1] - action[t] = 0
    # For the first hour: energy[0] - action[0] = 5.
    differences = diags(
        [np.ones(n), -np.ones(n - 1)],
        offsets=[0, -1],
        shape=(n, n),
        format="csc",
    )

    A_eq = hstack([-eye(n, format="csc"), differences], format="csc")
    b_eq = np.concatenate([[5.0], np.zeros(n - 1)])

    result = linprog(
        c=objective,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(-2.5, 2.5)] * n + [(0.0, 5.0)] * n,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(result.message)

    actions = result.x[:n]
    actions[np.abs(actions) < 1e-9] = 0.0

    energy = 5.0 + np.cumsum(actions)
    revenue = -prices * actions

    # Verify the resulting physical trajectory.
    assert np.all(energy >= -1e-7)
    assert np.all(energy <= 5.0 + 1e-7)
    assert np.all(np.abs(actions) <= 2.5 + 1e-7)

    # One concise output row per hour.
    output = df[["month", "day_of_month", "hour"]].copy()
    output["price_usd_per_mwh"] = prices
    output["action_mwh"] = actions
    output["energy_after_mwh"] = energy
    output["revenue_usd"] = revenue

    output_path = folder / f"optimal_{filename}"
    output.to_csv(output_path, index=False, float_format="%.6f")

    print(f"{filename}: optimal revenue = ${revenue.sum():,.2f}")
    print(f"Saved {output_path.name}\n")


if __name__ == "__main__":
    find_optimal_strategy("training_data.csv")
    find_optimal_strategy("initial_history.csv")