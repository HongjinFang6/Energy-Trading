"""Causal chronological experiment; MSE training remains separate from scheduling."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from baseline_core import fit_linear, predict_prices, optimize_schedule, energy_trajectory
from dataset import load_days, matrices


def simulate(days, prices, eta):
    energy = 5.0
    rows = []
    for d, predicted in zip(days, prices):
        start = perf_counter()
        p = optimize_schedule(predicted, energy, eta)
        duration = perf_counter()-start
        trajectory = energy_trajectory(p, energy, eta)
        if p.shape != (24,1) or not np.isfinite(p).all() or abs(p).max() > 2.5+1e-8:
            raise AssertionError("Invalid returned schedule")
        if trajectory.min() < -1e-8 or trajectory.max() > 5+1e-8:
            raise AssertionError("Energy bound violation")
        rows.append(dict(month=d["month"], day=d["day"], initial_energy=energy,
                         final_energy=float(trajectory[-1]), revenue=float(-d["y"]@p.ravel()),
                         solver_seconds=duration))
        energy = float(np.clip(trajectory[-1],0,5))
    return pd.DataFrame(rows)


def record(name, split, days, forecasts, eta, output):
    trace = simulate(days, forecasts, eta)
    trace.to_csv(output/f"daily_{split}_{name}_{eta:.2f}.csv", index=False)
    target = np.stack([d["y"] for d in days])
    return dict(model=name, split=split, eta=eta, days=len(days),
                price_mse=float(np.mean((forecasts-target)**2)),
                price_rmse=float(np.sqrt(np.mean((forecasts-target)**2))),
                price_mae=float(np.mean(abs(forecasts-target))),
                revenue=float(trace.revenue.sum()), violations=0,
                total_solver_seconds=float(trace.solver_seconds.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    days = load_days(args.data_dir)
    training = [d for d in days if d["month"] <= 6]
    validation = [d for d in days if 7 <= d["month"] <= 9]
    testing = [d for d in days if d["month"] >= 10]
    z, y = matrices(training)
    zv, yv = matrices(validation)
    # Optional extension, distinct from unregularized lecture baseline.
    selection = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = fit_linear(z, y, alpha)
        selection.append(dict(alpha=alpha, validation_mse=float(np.mean((predict_prices(model,zv)-yv)**2))))
    selected_alpha = min(selection,key=lambda r:r["validation_mse"])["alpha"]
    rows = []
    for split, fit_days, eval_days in [
        ("validation_Jul_Sep", training, validation),
        ("holdout_Oct_Dec", training+validation, testing),
    ]:
        zfit, yfit = matrices(fit_days)
        zeval, _ = matrices(eval_days)
        for name, alpha in [("ols",0.0),("ridge_optional",selected_alpha),("weekly_mean",None)]:
            if alpha is None:
                forecasts = np.stack([d["past_prices"].mean(axis=1) for d in eval_days])
            else:
                forecasts = predict_prices(fit_linear(zfit,yfit,alpha), zeval)
            for eta in [1.0,0.95]:
                result = record(name,split,eval_days,forecasts,eta,args.output_dir)
                rows.append(result)
                print(json.dumps(result),flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir/"metrics.csv",index=False)
    (args.output_dir/"ridge_selection.json").write_text(json.dumps(dict(
        selected_alpha=selected_alpha, selection=selection,
        criterion="July-September prediction MSE, not scheduling revenue",
        primary_baseline="OLS alpha=0; does not use Ridge selection"),indent=2))


if __name__ == "__main__":
    main()
