"""Construct daily (z_i, y_i) pairs, using only permitted inputs for z_i."""
from pathlib import Path
import numpy as np
import pandas as pd
from baseline_core import feature_vector

FORECAST = "da_net_load_forecast_mw"
LOAD = "realized_net_load_mw"
PRICE = "da_price_usd_per_mwh"
COLUMNS = ["month", "day_of_month", "day_of_week", "hour", FORECAST, LOAD, PRICE]


def load_days(data_dir):
    data_dir = Path(data_dir)
    training = pd.read_csv(data_dir/"training_data.csv")
    history = pd.read_csv(data_dir/"initial_history.csv")
    for frame in [training, history]:
        if set(frame.columns) != set(COLUMNS) or len(frame) % 24:
            raise ValueError("Unexpected CSV schema or incomplete daily block")
        if not np.isfinite(frame[COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite CSV observations")
        if not np.array_equal(frame.hour.to_numpy().reshape(-1, 24),
                              np.tile(np.arange(1, 25), (len(frame)//24, 1))):
            raise ValueError("Expected chronological hour blocks 1,...,24")
        metadata = frame[["month", "day_of_month", "day_of_week"]].to_numpy().reshape(-1,24,3)
        if not np.all(metadata == metadata[:, :1]):
            raise ValueError("Calendar changes inside a day")
    if len(history) != 168:
        raise ValueError("Expected seven initial historical days")
    full = pd.concat([history, training], ignore_index=True)
    arrays = {key: full[key].to_numpy().reshape(-1, 24) for key in [FORECAST, LOAD, PRICE]}
    days = []
    for i in range(7, len(full)//24):
        row = full.iloc[i*24]
        def past(key):
            return arrays[key][i-7:i][::-1].T.copy()
        record = dict(month=int(row.month), day=int(row.day_of_month),
                      weekday=int(row.day_of_week), forecast=arrays[FORECAST][i].copy(),
                      past_forecasts=past(FORECAST), past_load=past(LOAD),
                      past_prices=past(PRICE), y=arrays[PRICE][i].copy())
        record["z"] = feature_vector(record["month"], record["day"], record["weekday"],
            record["forecast"], record["past_forecasts"], record["past_load"], record["past_prices"])
        days.append(record)
    return days


def matrices(days):
    return np.stack([d["z"] for d in days]), np.stack([d["y"] for d in days])


def scheduling_arguments(day, initial_energy, days_remaining):
    return (day["month"], day["day"], day["weekday"], float(initial_energy),
            int(days_remaining), day["forecast"].reshape(24,1),
            day["past_forecasts"], day["past_load"], day["past_prices"])
