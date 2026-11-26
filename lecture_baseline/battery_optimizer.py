import numpy as np
from scipy.optimize import linprog


def optimal_schedule(predicted_prices, initial_energy=5.0):
    prices = np.asarray(predicted_prices).reshape(24)

    # L @ x gives cumulative energy changes after each hour.
    L = np.tril(np.ones((24, 24)))

    result = linprog(
        c=prices,
        A_ub=np.vstack([L, -L]),
        b_ub=np.concatenate([
            np.full(24, 5.0 - initial_energy),
            np.full(24, initial_energy),
        ]),
        bounds=[(-2.5, 2.5)] * 24,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x.reshape(24, 1)