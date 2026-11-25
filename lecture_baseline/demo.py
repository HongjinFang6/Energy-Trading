"""Call both required schedulers on the first supplied day and score afterwards."""
import argparse
import numpy as np
from baseline_core import load_model, predict_prices, energy_trajectory
from dataset import load_days, scheduling_arguments
from schedule_lossless import schedule_lossless
from schedule_lossy import schedule_lossy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    days = load_days(args.data_dir)
    d = days[0]
    predicted_y = predict_prices(load_model(),d["z"])
    print("Observed feature z shape:",d["z"].shape)
    print("Predicted price y_hat shape:",predicted_y.shape)
    print("Predicted prices:",np.round(predicted_y,2))
    for name, function, eta in [("lossless",schedule_lossless,1.0),("lossy",schedule_lossy,0.95)]:
        x = function(*scheduling_arguments(d,5.0,len(days)-1))
        energy = energy_trajectory(x,5.0,eta)
        revenue = -d["y"]@x.ravel()
        print(f"\n{name}: schedule x shape={x.shape}")
        print("Schedule (MWh):",np.round(x.ravel(),3))
        print("Stored energy (MWh):",np.round(energy,3))
        print(f"Realized revenue, scored after decision: ${revenue:.2f}")
    print("This uses the saved model; training-period revenue is not a held-out score.")


if __name__ == "__main__":
    main()
