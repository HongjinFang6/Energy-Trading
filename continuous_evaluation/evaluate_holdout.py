"""Chronological 80/20 holdout, with one continuous testing run per seed."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from battery import solve
from evaluate import load_data, evaluate_week, summarize
from model import Scheduler, train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 77, 123])
    parser.add_argument("--eta", type=float, choices=[1., .95], default=1.)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "holdout_results")
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    seeds = list(dict.fromkeys(args.seeds))
    market = load_data(args.data_dir)

    # Split complete days in chronological order. No shuffling.
    split = int(0.80 * market["days"])
    train_days = np.arange(split)
    test_days = np.arange(split, market["days"])
    def label(index):
        month, day, _ = market["calendar"][index + 7]
        return f"{month:02d}/{day:02d}"
    print(f"Training: {len(train_days)} days, {label(0)} to {label(split - 1)}")
    print(f"Testing:  {len(test_days)} days, {label(split)} to {label(market['days'] - 1)}")
    print("Testing starts at 5 MWh once, then carries energy across all testing days.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.csv").unlink(missing_ok=True)

    # One oracle covers the entire same test horizon, without daily resets.
    oracle = solve(market["price"][test_days + 7].ravel(), eta=args.eta)
    results, actions = [], []
    for seed in seeds:
        print(f"\nSeed {seed}", flush=True)
        model, mean, std = train_model(market, train_days, seed, args.epochs, args.lr)
        scheduler = Scheduler(model, mean, std, args.eta)
        # evaluate_week accepts any consecutive block, including all 74 test days.
        result, rows = evaluate_week(scheduler, market, test_days, args.eta)
        upper = oracle["revenue_upper_bound"]
        if upper is not None and result["revenue"] > upper + 1e-5:
            raise ValueError("Revenue exceeds oracle bound; check matching constraints")
        # 'week=1' identifies the single block for the shared summary helper.
        results.append(dict(
            seed=seed, eta=args.eta, week=1, start_day=split, days=len(test_days),
            **result, oracle_revenue=oracle["revenue"], oracle_upper_bound=upper,
            oracle_certified=oracle["certified_optimal"],
        ))
        for t, row in enumerate(rows):
            row.update(seed=seed, eta=args.eta,
                       oracle_action_mwh=oracle["schedule"][t, 0],
                       oracle_energy_after_mwh=oracle["energy"][t + 1])
        actions.extend(rows)
        pd.DataFrame(results).drop(columns="week").to_csv(args.output / "runs.csv", index=False)
        pd.DataFrame(actions).to_csv(args.output / "actions.csv", index=False)

    summary = summarize(pd.DataFrame(results), pd.DataFrame(actions), seeds,
                        args.eta, args.epochs, args.lr)
    summary = summary.rename(columns={"weeks": "test_runs", "failed_weeks": "failed_runs"})
    summary["training_days"] = len(train_days)
    summary["split"] = "chronological_80_20"
    summary.to_csv(args.output / "summary.csv", index=False)
    display = summary.round(2)
    display["learning_rate"] = summary["learning_rate"]
    print("\nChronological holdout results:")
    print(display.to_string(index=False))
    print("\nSaved summary.csv, runs.csv, actions.csv to", args.output)


if __name__ == "__main__":
    main()
