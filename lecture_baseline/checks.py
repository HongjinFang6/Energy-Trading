"""Independent tests for fitting, permitted inputs, physics, and required interfaces."""
import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from baseline_core import (feature_vector, fit_linear, predict_prices,
    optimize_schedule, energy_trajectory)
from dataset import load_days, scheduling_arguments


def assert_feasible(p, energy, eta):
    s = energy_trajectory(p,energy,eta)
    assert p.shape == (24,1) and np.isfinite(p).all()
    assert abs(p).max() <= 2.5+1e-8
    assert s.min() >= -1e-8 and s.max() <= 5+1e-8
    return float(np.clip(s[-1],0,5))


def exhaustive_oracle(prices, initial, eta):
    """Enumerate every charge/discharge mode, then independently solve its LP."""
    n = len(prices)
    lower = np.tril(np.ones((n,n)))
    best = np.inf
    for flags in product([False,True],repeat=n):
        slopes = np.where(flags,eta,1/eta)
        accumulated = lower*slopes[None,:]
        result = linprog(prices,A_ub=np.vstack([accumulated,-accumulated]),
            b_ub=np.r_[np.full(n,5-initial),np.full(n,initial)],
            bounds=[(0,2.5) if flag else (-2.5,0) for flag in flags],method="highs")
        if result.success:
            best = min(best,result.fun)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,default=Path("data"))
    args = parser.parse_args()
    rng = np.random.default_rng(77)
    # Known affine mapping is recovered by OLS, including intercept handling.
    z = rng.normal(size=(180,130))
    y = z@rng.normal(size=(130,24))+rng.normal(size=24)
    model = fit_linear(z,y)
    np.testing.assert_allclose(predict_prices(model,z),y,atol=1e-9)
    # Ridge satisfies the independent stationarity condition; intercept unpenalized.
    ridge = fit_linear(z,y,10)
    zs = (z-ridge["mean"])/ridge["scale"]
    residual = predict_prices(ridge,z)-y
    np.testing.assert_allclose(zs.T@residual+10*ridge["weights"],0,atol=1e-9)
    np.testing.assert_allclose(residual.mean(axis=0),0,atol=1e-10)
    for eta in [1.0,.95]:
        for _ in range(16):
            price = rng.normal(0,70,4)
            initial = rng.uniform(0,5)
            p = optimize_schedule(price,initial,eta).ravel()
            expected = exhaustive_oracle(price,initial,eta)
            np.testing.assert_allclose(price@p,expected,atol=1e-5,rtol=1e-7)
    # An initially full battery cannot charge in a single negative-price hour.
    np.testing.assert_allclose(optimize_schedule([-100],5,.95),[[0]],atol=1e-8)
    # No terminal target: ending full can be profitable with a negative last price.
    np.testing.assert_allclose(optimize_schedule([-100],0,.95),[[2.5]],atol=1e-8)
    days = load_days(args.data_dir)
    history = pd.read_csv(args.data_dir/"initial_history.csv")
    np.testing.assert_allclose(days[0]["past_prices"][:,0],history.iloc[-24:]["da_price_usd_per_mwh"])
    np.testing.assert_allclose(days[0]["past_prices"][:,6],history.iloc[:24]["da_price_usd_per_mwh"])
    # Current targets and realized loads cannot enter features, even if changed.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        data = pd.read_csv(args.data_dir/"training_data.csv")
        data.loc[:23,"da_price_usd_per_mwh"] = 1e9
        data.loc[:23,"realized_net_load_mw"] = -1e9
        data.to_csv(root/"training_data.csv",index=False)
        history.to_csv(root/"initial_history.csv",index=False)
        changed = load_days(root)
        np.testing.assert_array_equal(changed[0]["z"],days[0]["z"])
        assert not np.array_equal(changed[0]["y"],days[0]["y"])
        assert not np.array_equal(changed[1]["z"],days[1]["z"])
    from schedule_lossless import schedule_lossless
    from schedule_lossy import schedule_lossy
    timings = {}
    for function,eta in [(schedule_lossless,1.0),(schedule_lossy,.95)]:
        energy = 5.0
        start = perf_counter()
        for j,d in enumerate(days):
            p = function(*scheduling_arguments(d,energy,len(days)-j-1))
            energy = assert_feasible(p,energy,eta)
        timings[str(eta)] = perf_counter()-start
        for initial in [0,1e-8,.37,2.631578947,4.99999999,5]:
            p = function(*scheduling_arguments(days[0],initial,0))
            assert_feasible(p,initial,eta)
    # Verify that each entry module imports without importing the other one.
    for name in ["schedule_lossless","schedule_lossy"]:
        other = "schedule_lossy" if name == "schedule_lossless" else "schedule_lossless"
        code = f"import sys; import {name}; assert {other!r} not in sys.modules"
        subprocess.run([sys.executable,"-c",code],cwd=Path(__file__).parent,check=True)
    print(json.dumps(dict(status="passed",optimizer_instances_vs_exhaustive=32,
        chronological_interface_calls=2*len(days),additional_fractional_state_calls=12,
        interface_seconds=timings),indent=2))


if __name__ == "__main__":
    main()
