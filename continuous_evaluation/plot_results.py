"""Plot chronological-holdout prices and cumulative realized trading revenue."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


def load_results(folder):
    summary = pd.read_csv(folder / "summary.csv").iloc[0]
    if str(summary["eligible"]).lower() != "true":
        raise ValueError(f"{folder}: incomplete/failed experiment")
    if "split" not in summary or summary["split"] != "chronological_80_20":
        raise ValueError("Use chronological holdout results, not disconnected weekly CV blocks")
    data = pd.read_csv(folder / "actions.csv")
    if data.duplicated(["seed", "day_index", "hour"]).any():
        raise ValueError("Duplicate hourly records")
    groups = [g.sort_values(["day_index", "hour"]).reset_index(drop=True)
              for _, g in data.groupby("seed")]
    first = groups[0]
    expected_hours = int(summary["days"]) * 24
    if len(first) != expected_hours:
        raise ValueError("Incomplete hourly records")
    for g in groups:
        pd.testing.assert_frame_equal(
            g[["day_index", "hour", "actual_price", "eta"]],
            first[["day_index", "hour", "actual_price", "eta"]],
        )
    predicted = np.stack([g.predicted_price.to_numpy() for g in groups])
    # Accumulate each seed's actual schedule revenue BEFORE averaging seeds.
    profits = np.stack([np.r_[0., g.revenue_usd.cumsum()] for g in groups])
    oracle = np.r_[0., (-first.actual_price * first.oracle_action_mwh).cumsum()]
    np.testing.assert_allclose(profits[:, -1].mean(), summary.mean_revenue, atol=1e-5)
    np.testing.assert_allclose(oracle[-1], summary.oracle_revenue, atol=1e-5)
    return first, predicted, profits, oracle, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("holdout_terminal"))
    parser.add_argument("--baseline", type=Path, help="Optional original holdout_results folder")
    parser.add_argument("--day", help="Day to zoom in on, e.g. 10/19; default is first test day")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    data, predicted, profits, oracle, summary = load_results(args.results)
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), layout="constrained")
    x = np.arange(len(data)) / 24
    end_times = np.arange(len(data) + 1) / 24
    mean_price = predicted.mean(axis=0)
    axes[0].plot(x, data.actual_price, label="Actual prices (perfect-price input)", linewidth=1)
    axes[0].plot(x, mean_price, label="MLP prediction, mean across seeds", linewidth=1, alpha=.8)
    axes[0].set(title="Prices across the complete testing period", ylabel="Price ($/MWh)")

    day_label = args.day or f"{int(data.month.iloc[0]):02d}/{int(data.day.iloc[0]):02d}"
    month, day = map(int, day_label.split("/"))
    mask = ((data.month == month) & (data.day == day)).to_numpy()
    if mask.sum() != 24:
        raise ValueError(f"{day_label} must identify exactly one testing day")
    hours = data.loc[mask, "hour"]
    axes[1].plot(hours, data.loc[mask, "actual_price"], marker="o", label="Actual prices")
    axes[1].plot(hours, mean_price[mask], marker="o", label="MLP prediction, mean across seeds")
    axes[1].fill_between(hours, predicted[:, mask].min(axis=0), predicted[:, mask].max(axis=0),
                         alpha=.15, label="Range across seeds")
    axes[1].set(title=f"Hourly price detail: {month:02d}/{day:02d}",
                xlabel="Hour", ylabel="Price ($/MWh)")

    mean_profit = profits.mean(axis=0)
    axes[2].plot(end_times, mean_profit,
                 label=f"{args.results.name}: mean final ${mean_profit[-1]:,.2f}")
    axes[2].fill_between(end_times, profits.min(axis=0), profits.max(axis=0),
                         alpha=.15, label="Range across seeds")
    certified = str(summary.oracle_certified).lower() == "true"
    label = "Perfect-price optimum" if certified else "Perfect-price feasible benchmark"
    axes[2].plot(end_times, oracle, color="black", linestyle="--",
                 label=f"{label}: final ${oracle[-1]:,.2f}")
    if args.baseline:
        previous, _, baseline_profit, _, _ = load_results(args.baseline)
        pd.testing.assert_frame_equal(
            previous[["day_index", "hour", "actual_price", "eta"]],
            data[["day_index", "hour", "actual_price", "eta"]],
        )
        curve = baseline_profit.mean(axis=0)
        axes[2].plot(end_times, curve,
                     label=f"{args.baseline.name}: mean final ${curve[-1]:,.2f}")
    axes[2].set(title="Cumulative realized revenue (terminal credits excluded)", ylabel="Revenue ($)")
    axes[2].yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    daily = data.iloc[::24]
    tick_indices = np.unique(np.linspace(0, len(daily) - 1, min(8, len(daily))).astype(int))
    labels = [f"{int(daily.iloc[i].month):02d}/{int(daily.iloc[i].day):02d}" for i in tick_indices]
    for ax in [axes[0], axes[2]]:
        ax.set_xticks(tick_indices, labels)
        ax.set_xlabel("Testing date")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    output = args.results / "comparison.png"
    figure.savefig(output, dpi=200)
    print("Saved", output)
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
