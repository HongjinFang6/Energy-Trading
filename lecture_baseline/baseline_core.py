"""Lecture baseline: z -> MSE-trained linear price vector -> feasible schedule.

No model training, CSV access, or network access occurs in a scheduling call.
"""
from functools import lru_cache
from pathlib import Path
import warnings
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint

HOURS = 24
CAPACITY = 5.0
POWER = 2.5


def feature_vector(month, day, weekday, net_load_forecast,
                   past_net_load_forecasts, past_realized_net_load,
                   past_realized_da_prices):
    """One day -> z in R^130; historical column zero is yesterday.

    Blocks: current forecast (24), yesterday prices (24), week-ago prices (24),
    weekly same-hour mean prices (24), historical mean net-load errors (24),
    seasonal sine/cosine + day-of-month (3), weekday indicators (7).
    """
    forecast = np.asarray(net_load_forecast, dtype=float).reshape(24)
    past_forecasts = np.asarray(past_net_load_forecasts, dtype=float)
    past_load = np.asarray(past_realized_net_load, dtype=float)
    past_prices = np.asarray(past_realized_da_prices, dtype=float)
    if any(a.shape != (24, 7) for a in [past_forecasts, past_load, past_prices]):
        raise ValueError("Historical arrays must have shape (24, 7)")
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1 <= weekday <= 7):
        raise ValueError("Invalid calendar input")
    angle = 2*np.pi*((month-1)+(day-1)/31)/12
    z = np.concatenate([
        forecast/10000,
        past_prices[:, 0],
        past_prices[:, 6],
        past_prices.mean(axis=1),
        (past_load-past_forecasts).mean(axis=1)/10000,
        [np.sin(angle), np.cos(angle), day/31],
        np.eye(7)[int(weekday)-1],
    ])
    if z.shape != (130,) or not np.isfinite(z).all():
        raise ValueError("Invalid feature vector")
    return z


def fit_linear(z, y, alpha=0.0):
    """Fit g(z)=Az+b. alpha=0 is the exact unregularized MSE baseline.

    For alpha>0, solve ||Z_standardized W - Y_centered||_F^2 + alpha||W||_F^2.
    This optional Ridge extension is distinguished from the lecture baseline.
    All feature normalization is fitted on the supplied training rows only.
    """
    z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    if z.ndim != 2 or y.shape != (len(z), 24) or alpha < 0:
        raise ValueError("Expected Z=(days, features), Y=(days, 24), alpha>=0")
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite training observations")
    mean = z.mean(axis=0)
    scale = z.std(axis=0)
    scale[scale < 1e-12] = 1.0
    zs = (z-mean)/scale
    intercept = y.mean(axis=0)
    centered_y = y-intercept
    if alpha == 0:
        # SVD-based least squares also handles non-unique/rank-deficient solutions.
        weights = np.linalg.lstsq(zs, centered_y, rcond=None)[0]
    else:
        weights = np.linalg.solve(zs.T@zs + alpha*np.eye(zs.shape[1]), zs.T@centered_y)
    return {"mean":mean, "scale":scale, "weights":weights,
            "intercept":intercept, "alpha":np.array(float(alpha)),
            "feature_version":np.array(1), "training_days":np.array(len(z))}


def predict_prices(model, z):
    z = np.asarray(z, dtype=float)
    predicted = ((z-model["mean"])/model["scale"])@model["weights"] + model["intercept"]
    if predicted.shape[-1] != 24 or not np.isfinite(predicted).all():
        raise ValueError("Invalid predicted price vector")
    return predicted


def save_model(model, path):
    np.savez_compressed(path, **model)


@lru_cache(maxsize=1)
def load_model():
    path = Path(__file__).resolve().with_name("price_model.npz")
    if not path.is_file():
        raise FileNotFoundError("Run python train.py --data-dir data first")
    with np.load(path, allow_pickle=False) as data:
        model = {key:data[key].copy() for key in data.files}
    if int(model["feature_version"]) != 1 or model["weights"].shape != (130, 24):
        raise ValueError("Model has incompatible feature schema")
    return model


def energy_trajectory(schedule, initial_energy, eta=1.0):
    p = np.asarray(schedule, dtype=float).ravel()
    delta = eta*np.maximum(p, 0)+np.minimum(p, 0)/eta
    return np.r_[initial_energy, initial_energy+np.cumsum(delta)]


def physical_clip(schedule, initial_energy, eta):
    """Enforce physical bounds sequentially, including solver roundoff repair.

    This is not a Euclidean projection or a learned direct policy. It changes
    only the returned optimization schedule and simulates the original dynamics.
    """
    energy = float(initial_energy)
    result = []
    for value in np.asarray(schedule).ravel():
        lower = max(-POWER, -eta*energy)
        upper = min(POWER, (CAPACITY-energy)/eta)
        p = float(np.clip(value, lower, upper))
        result.append(p)
        energy += eta*max(p, 0)+min(p, 0)/eta
        energy = float(np.clip(energy, 0, CAPACITY))
    return np.asarray(result).reshape(-1, 1)


def optimize_schedule(predicted_prices, initial_energy, eta=1.0):
    """Certainty-equivalent daily cost minimization. No terminal value/target.

    Lossless: LP in the net grid exchanges p.
    Lossy: exact physical MILP in c,d,end-of-hour states s,binary charge modes u.
    A finite feasible solver incumbent is accepted on an early stop; otherwise
    return idle. Solver time limits do not imply certified optimality.
    """
    prices = np.asarray(predicted_prices, dtype=float).ravel()
    n = len(prices)
    if n < 1 or not np.isfinite(prices).all() or not 0 <= initial_energy <= CAPACITY:
        raise ValueError("Invalid optimizer input")
    if eta not in (1.0, 0.95):
        raise ValueError("Project supports eta=1 or 0.95")
    if eta == 1.0:
        cumulative = np.tril(np.ones((n, n)))
        result = linprog(
            c=prices,
            A_ub=np.vstack([cumulative, -cumulative]),
            b_ub=np.r_[np.full(n, CAPACITY-initial_energy), np.full(n, initial_energy)],
            bounds=[(-POWER, POWER)]*n,
            method="highs", options={"time_limit":5.0})
        candidate = result.x
    else:
        identity = np.eye(n)
        zero = np.zeros((n, n))
        differences = identity-np.eye(n, k=-1)
        # s[t] - s[t-1] - eta*c[t] + d[t]/eta = 0 (initial energy in RHS).
        dynamics = np.hstack([-eta*identity, identity/eta, differences, zero])
        # c <= POWER*u and d <= POWER*(1-u).
        charge_mode = np.hstack([identity, zero, zero, -POWER*identity])
        discharge_mode = np.hstack([zero, identity, zero, POWER*identity])
        rhs = np.r_[initial_energy, np.zeros(n-1)]
        constraints = LinearConstraint(
            np.vstack([dynamics, charge_mode, discharge_mode]),
            np.r_[rhs, np.full(2*n, -np.inf)],
            np.r_[rhs, np.zeros(n), np.full(n, POWER)])
        result = milp(
            c=np.r_[prices, -prices, np.zeros(2*n)],
            integrality=np.r_[np.zeros(3*n), np.ones(n)],
            bounds=Bounds(np.zeros(4*n),
                          np.r_[np.full(2*n, POWER), np.full(n, CAPACITY), np.ones(n)]),
            constraints=constraints,
            options={"time_limit":5.0, "mip_rel_gap":1e-8})
        candidate = None if result.x is None else result.x[:n]-result.x[n:2*n]
    if candidate is None or not np.isfinite(candidate).all():
        warnings.warn("Optimizer has no incumbent; using a feasible idle schedule")
        return np.zeros((n, 1))
    if not result.success:
        warnings.warn("Optimizer stopped early; using a physically repaired incumbent")
    return physical_clip(candidate, initial_energy, eta)


def schedule_day(month, day, weekday, initial_energy, days_remaining,
                 net_load_forecast, past_net_load_forecasts,
                 past_realized_net_load, past_realized_da_prices, eta):
    # days_remaining is intentionally unused in this myopic daily baseline.
    z = feature_vector(month, day, weekday, net_load_forecast,
                       past_net_load_forecasts, past_realized_net_load,
                       past_realized_da_prices)
    predicted_y = predict_prices(load_model(), z)
    return optimize_schedule(predicted_y, float(initial_energy), eta)
