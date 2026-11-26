from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from battery_optimizer import optimal_schedule
from g import MLP

torch.manual_seed(42)

# --------------------------------------------------
# 1. Load data and construct features
# --------------------------------------------------

folder = Path(__file__).resolve().parent  # perfect_foresight/
data_folder = folder.parent              # energy-storage-project/
df = pd.read_csv(data_folder / "training_data.csv")
days = df.iloc[::24]

# One-hot encode month, day of month, and weekday.
month = np.eye(12)[days["month"].to_numpy(dtype=int) - 1]
day_of_month = np.eye(31)[days["day_of_month"].to_numpy(dtype=int) - 1]
weekday = np.eye(7)[days["day_of_week"].to_numpy(dtype=int) - 1]

calendar = np.column_stack([month, day_of_month, weekday])

# One-hot encode hour.
hour = np.eye(24)[df["hour"].to_numpy(dtype=int) - 1]
hour = hour.reshape(-1, 24, 24)

# Daily forecasts and target prices.
forecasts = df["da_net_load_forecast_mw"].to_numpy().reshape(-1, 24)
Y = df["da_price_usd_per_mwh"].to_numpy().reshape(-1, 24)

# Random 80/20 split of complete days.
train_days, test_days = train_test_split(
    np.arange(len(days)),
    test_size=0.20,
    random_state=42,
    shuffle=True,
)

# Standardize only forecasts, using training statistics.
mean = forecasts[train_days].mean(axis=0)
std = forecasts[train_days].std(axis=0)
std[std == 0] = 1

scaled_forecasts = (forecasts - mean) / std

# Each hour receives the calendar, all 24 forecasts, and its hour indicator.
daily_features = np.column_stack([calendar, scaled_forecasts])
repeated_features = np.repeat(daily_features[:, None, :], 24, axis=1)
Z = np.concatenate([repeated_features, hour], axis=2)

z_train = torch.tensor(Z[train_days], dtype=torch.float32)
z_test = torch.tensor(Z[test_days], dtype=torch.float32)
y_train = torch.tensor(Y[train_days], dtype=torch.float32)
y_test = Y[test_days]

# --------------------------------------------------
# 2. Train the price predictor g_theta
# --------------------------------------------------

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(2000):
    predicted = model(z_train)

    # (1/K) * sum_i ||y_i - g_theta(z_i)||²
    loss = ((y_train - predicted) ** 2).sum(dim=1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}: training loss = {loss.item():.2f}")

# --------------------------------------------------
# 3. Evaluate price predictions
# --------------------------------------------------

model.eval()

with torch.no_grad():
    predictions = model(z_test).numpy()

errors = predictions - y_test

print(f"\nTraining days: {len(train_days)}")
print(f"Testing days:  {len(test_days)}")
print(f"Test MAE:  ${np.abs(errors).mean():.2f}/MWh")
print(f"Test RMSE: ${np.sqrt((errors**2).mean()):.2f}/MWh")

# --------------------------------------------------
# 4. Optimize one held-out day's battery schedule
# --------------------------------------------------

# Select the first day in the randomly ordered test set.
day_index = 0
selected_day = days.iloc[test_days[day_index]]
day_label = (
    f"{int(selected_day['month']):02d}/"
    f"{int(selected_day['day_of_month']):02d}"
)

predicted_prices = predictions[day_index]
actual_prices = y_test[day_index]

initial_energy = 5.0

# x*(g_theta(z)): use predicted prices to choose actions.
schedule = optimal_schedule(
    predicted_prices,
    initial_energy,
).ravel()

# Lossless energy trajectory: starting state plus 24 ending states.
energy = np.concatenate([
    [initial_energy],
    initial_energy + np.cumsum(schedule),
])

# Score the SAME schedule at predicted and actual prices.
predicted_revenue = -predicted_prices @ schedule
actual_revenue = -actual_prices @ schedule

print(f"\nSelected test day: {day_label} (MM/DD)")
print("Battery schedule (MWh):", schedule.round(3))
print("Stored energy (MWh):  ", energy.round(3))
print(f"Predicted revenue: ${predicted_revenue:.2f}")
print(f"Actual revenue:    ${actual_revenue:.2f}")

# --------------------------------------------------
# 5. Plot prediction performance
# --------------------------------------------------

hours = np.arange(1, 25)

fig_prices, axes = plt.subplots(
    1, 3, figsize=(16, 4), layout="constrained"
)

axes[0].plot(hours, actual_prices, label="Actual")
axes[0].plot(hours, predicted_prices, label="Predicted")
axes[0].set(
    title=f"Prices on {day_label}",
    xlabel="Hour",
    ylabel="DA price ($/MWh)",
)
axes[0].legend()

axes[1].bar(hours, errors[day_index])
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(
    title=f"Prediction errors on {day_label}",
    xlabel="Hour",
    ylabel="Predicted − actual ($/MWh)",
)

axes[2].hist(errors.ravel(), bins=40, edgecolor="white")
axes[2].axvline(0, color="black", linestyle="--")
axes[2].set(
    title="Errors across all testing days",
    xlabel="Predicted − actual ($/MWh)",
    ylabel="Count",
)

fig_prices.savefig(folder / "prediction_plots.png", dpi=200)

# --------------------------------------------------
# 6. Plot the battery schedule and stored energy
# --------------------------------------------------

fig_battery, axes_battery = plt.subplots(
    2, 1, figsize=(10, 7), layout="constrained"
)

colors = np.where(schedule >= 0, "tab:blue", "tab:orange")

axes_battery[0].bar(hours, schedule, color=colors)
axes_battery[0].axhline(0, color="black", linewidth=1)
axes_battery[0].set(
    title=f"Schedule on {day_label}: positive = charge, negative = discharge",
    xlabel="Hour",
    ylabel="Grid exchange (MWh)",
    ylim=(-3, 3),
)

axes_battery[1].step(
    np.arange(25), energy, where="post", marker="o"
)
axes_battery[1].axhline(0, color="gray", linestyle="--")
axes_battery[1].axhline(5, color="gray", linestyle="--")
axes_battery[1].set(
    title=f"Stored energy — actual daily revenue: ${actual_revenue:.2f}",
    xlabel="Hour boundary (0 = start of day)",
    ylabel="Stored energy (MWh)",
    ylim=(-0.2, 5.2),
)

fig_battery.savefig(folder / "battery_schedule.png", dpi=200)

# Display both figures after creating and saving them.
plt.show()