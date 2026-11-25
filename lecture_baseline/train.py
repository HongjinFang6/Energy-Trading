"""Offline MSE price-model fitting. Default alpha=0 matches the lecture exactly."""
import argparse
from pathlib import Path
import numpy as np
from baseline_core import fit_linear, predict_prices, save_model
from dataset import load_days, matrices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="0 = lecture OLS baseline; positive = optional Ridge extension")
    parser.add_argument("--through-month", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("price_model.npz"))
    args = parser.parse_args()
    days = [d for d in load_days(args.data_dir) if d["month"] <= args.through_month]
    z, y = matrices(days)
    model = fit_linear(z, y, args.alpha)
    save_model(model, args.output)
    mse = np.mean((predict_prices(model, z)-y)**2)
    print(f"Fitted {len(days)} daily pairs: Z={z.shape}, Y={y.shape}, alpha={args.alpha:g}")
    print(f"Training MSE={mse:.6f}; this is an in-sample diagnostic")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
