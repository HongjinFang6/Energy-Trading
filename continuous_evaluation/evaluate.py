"""Run five-fold weekly-block CV. Usage: python evaluate.py"""
import argparse
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from battery import check_schedule, solve
from model import Scheduler, train_model


def load_data(folder):
    columns = ["month", "day_of_month", "day_of_week", "hour",
               "da_net_load_forecast_mw", "realized_net_load_mw", "da_price_usd_per_mwh"]
    history = pd.read_csv(folder / "initial_history.csv")[columns]
    training = pd.read_csv(folder / "training_data.csv")[columns]
    if len(history) != 168 or len(training) < 35 * 24 or len(training) % 24:
        raise ValueError("Expected seven history days and at least five full training weeks")
    data = pd.concat([history, training], ignore_index=True)
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Data contain missing or non-finite values")
    if not np.all(data.hour.to_numpy().reshape(-1, 24) == np.arange(1, 25)):
        raise ValueError("Each day must contain hours 1..24 in order")
    meta = data[columns[:3]].to_numpy().reshape(-1, 24, 3)
    if not np.all(meta == meta[:, :1]) or not np.all(meta == np.floor(meta)):
        raise ValueError("Invalid calendar information within a day")
    calendar = meta[:, 0].astype(int)
    if not np.all((calendar[:, 2] >= 1) & (calendar[:, 2] <= 7)):
        raise ValueError("Weekday must be in 1..7")
    # The CSV has no year. Accept a globally consecutive leap or non-leap
    # calendar, including the December-to-January history boundary.
    def consecutive(start_year):
        try:
            current = date(start_year, *calendar[0, :2])
            for row in calendar:
                if (current.month, current.day) != tuple(row[:2]):
                    return False
                current += timedelta(days=1)
            return True
        except ValueError:
            return False
    if not any(consecutive(year) for year in range(1999, 2003)):
        raise ValueError("Dates must be consecutive without gaps")
    if not np.all(calendar[1:, 2] == calendar[:-1, 2] % 7 + 1):
        raise ValueError("Weekdays must advance consecutively")
    return dict(
        calendar=calendar, days=len(training) // 24,
        forecast=data[columns[4]].to_numpy(dtype=float).reshape(-1, 24),
        load=data[columns[5]].to_numpy(dtype=float).reshape(-1, 24),
        price=data[columns[6]].to_numpy(dtype=float).reshape(-1, 24),
    )


def day_inputs(market, index, energy, days_remaining):
    k = int(index) + 7
    month, day, weekday = market["calendar"][k]
    def past(key):
        return market[key][k - 7:k][::-1].T.copy()
    return (int(month), int(day), int(weekday), float(energy), int(days_remaining),
            market["forecast"][k].reshape(24, 1).copy(),
            past("forecast"), past("load"), past("price"))


def evaluate_week(scheduler, market, indices, eta):
    """Start full ONCE; reject invalid schedules and carry energy between days."""
    energy, runtime, revenue = 5., 0., 0.
    rows = []
    for j, index in enumerate(indices):
        try:
            inputs = day_inputs(market, index, energy, len(indices) - j - 1)
            before = perf_counter()
            actions = scheduler(*inputs)
            runtime += perf_counter() - before
            if runtime > 30 * len(indices):
                raise TimeoutError("Exceeded 30 * days scheduler seconds")
            states = check_schedule(actions, energy, eta)
            predicted = scheduler.prices(inputs[0], inputs[1], inputs[2], inputs[5])
            # Actual prices are accessed only after choosing the day's actions.
            actual = market["price"][index + 7]
            p = np.asarray(actions).ravel()
            revenue += float(-actual @ p)
            month, day, _ = market["calendar"][index + 7]
            for h in range(24):
                rows.append(dict(
                    day_index=int(index), month=int(month), day=int(day), hour=h + 1,
                    actual_price=actual[h], predicted_price=predicted[h],
                    action_mwh=p[h], energy_before_mwh=states[h],
                    energy_after_mwh=states[h + 1], revenue_usd=-actual[h] * p[h],
                ))
            energy = float(np.clip(states[-1], 0, 5))
        except Exception as error:
            return dict(status="failed", revenue=np.nan, seconds=runtime,
                        error=f"day {index}: {type(error).__name__}: {error}"), rows
    return dict(status="passed", revenue=revenue, seconds=runtime, error=""), rows


def summarize(weekly, actions, seeds, eta, epochs, learning_rate):
    totals = weekly.groupby("seed").agg(
        revenue=("revenue", "sum"), passed=("status", lambda x: (x == "passed").all()))
    totals.loc[~totals.passed, "revenue"] = np.nan
    valid = totals.loc[totals.passed, "revenue"]
    oracle_rows = weekly.drop_duplicates("week")
    oracle_revenue = oracle_rows.oracle_revenue.sum()
    upper = (oracle_rows.oracle_upper_bound.sum()
             if oracle_rows.oracle_upper_bound.notna().all() else np.nan)
    certified = bool(oracle_rows.oracle_certified.all())
    errors = actions.predicted_price - actions.actual_price if len(actions) else pd.Series(dtype=float)
    eligible = bool(totals.passed.all() and len(totals) == len(seeds))
    # A failed seed makes the overall experiment ineligible; do not report
    # a deceptively good mean that silently discards failed seeds.
    average = valid.mean() if eligible else np.nan
    return pd.DataFrame([dict(
        model="MLP", eta=eta, epochs=epochs, learning_rate=learning_rate,
        seeds=",".join(map(str, seeds)), weeks=len(oracle_rows),
        days=int(oracle_rows.days.sum()), eligible=eligible,
        failed_weeks=int((weekly.status != "passed").sum()),
        mean_revenue=average,
        std_revenue=valid.std() if eligible and len(valid) > 1 else np.nan,
        worst_seed_revenue=valid.min() if eligible else np.nan,
        oracle_revenue=oracle_revenue, oracle_upper_bound=upper,
        oracle_certified=certified,
        oracle_capture_pct=100 * average / oracle_revenue
            if certified and oracle_revenue > 0 else np.nan,
        mean_regret=oracle_revenue - average if certified else np.nan,
        price_mae=errors.abs().mean() if eligible else np.nan,
        price_rmse=np.sqrt((errors ** 2).mean()) if eligible else np.nan,
    )])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 77, 123])
    parser.add_argument("--eta", type=float, choices=[1., .95], default=1.)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    seeds = list(dict.fromkeys(args.seeds))
    market = load_data(args.data_dir)
    days = np.arange(market["days"])
    weeks = [days[start:start + 7] for start in range(0, len(days), 7)]
    args.output.mkdir(parents=True, exist_ok=True)
    # An interrupted rerun must not leave an old summary looking current.
    (args.output / "summary.csv").unlink(missing_ok=True)
    weekly_rows, action_rows = [], []
    evaluated = []
    for fold in range(5):
        held_out = [w for w in range(len(weeks)) if w % 5 == fold]
        train_days = np.concatenate([week for w, week in enumerate(weeks) if w % 5 != fold])
        validation = np.concatenate([weeks[w] for w in held_out])
        assert not np.intersect1d(train_days, validation).size
        evaluated.extend(validation.tolist())
        print(f"\nFold {fold + 1}/5: {len(train_days)} fitting days, "
              f"{len(validation)} held-out days", flush=True)
        # The same continuous-week oracle is shared by all random seeds.
        oracles = {w: solve(market["price"][weeks[w] + 7].ravel(), eta=args.eta)
                   for w in held_out}
        for seed in seeds:
            print(f"Seed {seed}", flush=True)
            model, mean, std = train_model(market, train_days, seed, args.epochs, args.lr)
            scheduler = Scheduler(model, mean, std, args.eta)
            for w in held_out:
                result, rows = evaluate_week(scheduler, market, weeks[w], args.eta)
                oracle = oracles[w]
                upper = oracle["revenue_upper_bound"]
                if upper is not None and result["revenue"] > upper + 1e-5:
                    raise ValueError("Revenue exceeds oracle bound; check matching constraints")
                weekly_rows.append(dict(
                    fold=fold + 1, week=w + 1, seed=seed, eta=args.eta,
                    start_day=int(weeks[w][0]), days=len(weeks[w]), **result,
                    oracle_revenue=oracle["revenue"], oracle_upper_bound=upper,
                    oracle_certified=oracle["certified_optimal"],
                ))
                for t, row in enumerate(rows):
                    row.update(fold=fold + 1, week=w + 1, seed=seed, eta=args.eta,
                               oracle_action_mwh=oracle["schedule"][t, 0],
                               oracle_energy_after_mwh=oracle["energy"][t + 1])
                action_rows.extend(rows)
            # Save progress in the same two files; no per-fold output folders.
            pd.DataFrame(weekly_rows).to_csv(args.output / "weekly.csv", index=False)
            pd.DataFrame(action_rows).to_csv(args.output / "actions.csv", index=False)
    np.testing.assert_array_equal(np.sort(evaluated), days)
    weekly, actions = pd.DataFrame(weekly_rows), pd.DataFrame(action_rows)
    summary = summarize(weekly, actions, seeds, args.eta, args.epochs, args.lr)
    summary.to_csv(args.output / "summary.csv", index=False)
    print("\nWeekly-block cross-validation (not a forward-time test):")
    display = summary.round(2)
    display["learning_rate"] = summary["learning_rate"]
    print(display.to_string(index=False))
    print("\nSaved summary.csv, weekly.csv, actions.csv to", args.output)


if __name__ == "__main__":
    main()
