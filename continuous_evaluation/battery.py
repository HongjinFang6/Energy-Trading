"""Shared physical model; sparse full-horizon LP/MILP and certified bound reporting."""
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import eye, diags, hstack, vstack, csc_matrix


def trajectory(p, initial_energy, eta):
    p = np.asarray(p, dtype=float).ravel()
    return np.r_[initial_energy, initial_energy + np.cumsum(
        eta*np.maximum(p, 0)+np.minimum(p, 0)/eta)]


def check_schedule(p, initial_energy, eta, expected_hours=24, tolerance=1e-7):
    p = np.asarray(p, dtype=float)
    if p.shape != (expected_hours, 1) or not np.isfinite(p).all():
        raise ValueError(f"Expected finite ({expected_hours},1) output, got {p.shape}")
    s = trajectory(p, initial_energy, eta)
    if abs(p).max() > 2.5+tolerance or s.min() < -tolerance or s.max() > 5+tolerance:
        raise ValueError("Power or energy constraint violation")
    return s


def solve(prices, initial_energy=5.0, eta=1.0, time_limit=60.0, terminal_value=0.0):
    """Maximize trading revenue + terminal_value * final stored energy.

    eta=.95 uses binary modes: a physically exact formulation at negative prices.
    A time-limited feasible solution is an incumbent, NOT a proven optimum.
    The bound is for this augmented objective. With terminal_value=0 (the
    default used by the perfect-price oracle), it is a cash-revenue bound.
    """
    price = np.asarray(prices, dtype=float).ravel()
    n = len(price)
    if n < 1 or not np.isfinite(price).all() or not 0 <= initial_energy <= 5:
        raise ValueError("Invalid horizon/prices/initial energy")
    if eta not in (1.0, .95):
        raise ValueError("eta must be 1 or .95")
    if not np.isfinite(terminal_value) or terminal_value < 0:
        raise ValueError("terminal_value must be finite and nonnegative")
    I = eye(n, format="csc")
    D = diags([np.ones(n), -np.ones(n-1)], [0,-1], shape=(n,n), format="csc")
    rhs = np.r_[initial_energy, np.zeros(n-1)]
    if eta == 1:
        # Variables [net exchanges p, end-of-hour stored energy s].
        low, high = np.r_[np.full(n,-2.5),np.zeros(n)], np.r_[np.full(n,2.5),np.full(n,5.)]
        objective = np.r_[price,np.zeros(n)]
        objective[2*n-1] = -terminal_value  # Last stored-energy variable.
        result = linprog(objective,
            A_eq=hstack([-I,D],format="csc"),b_eq=rhs,
            bounds=np.column_stack([low,high]),method="highs",
            options={"time_limit":time_limit})
        p = None if result.x is None else result.x[:n]
        dual_cost = None
        if result.success:
            dual_cost = float(rhs@result.eqlin.marginals +
                low@result.lower.marginals + high@result.upper.marginals)
    else:
        # Variables [charge c, discharge d, energy s, binary charge mode u].
        Z = csc_matrix((n,n))
        dynamics = hstack([-.95*I,I/.95,D,Z],format="csc")
        mode_c = hstack([I,Z,Z,-2.5*I],format="csc")
        mode_d = hstack([Z,I,Z,2.5*I],format="csc")
        constraint = LinearConstraint(vstack([dynamics,mode_c,mode_d],format="csc"),
            np.r_[rhs,np.full(2*n,-np.inf)],np.r_[rhs,np.zeros(n),np.full(n,2.5)])
        objective = np.r_[price,-price,np.zeros(2*n)]
        objective[3*n-1] = -terminal_value  # Last stored-energy variable.
        result = milp(objective,
            integrality=np.r_[np.zeros(3*n),np.ones(n)],
            bounds=Bounds(np.zeros(4*n),np.r_[np.full(2*n,2.5),np.full(n,5.),np.ones(n)]),
            constraints=constraint,options={"time_limit":time_limit,"mip_rel_gap":1e-8})
        p = None if result.x is None else result.x[:n]-result.x[n:2*n]
        dual_cost = getattr(result,"mip_dual_bound",None)
    if p is None or not np.isfinite(p).all():
        raise RuntimeError(f"No usable solver solution: {result.message}")
    p = p.reshape(-1,1)
    # Check the original net-exchange physics, not just internal LP/MILP states.
    s = check_schedule(p,initial_energy,eta,n,tolerance=1e-6)
    revenue = float(-price@p.ravel())
    objective_value = revenue + terminal_value * float(s[-1])
    upper = -float(dual_cost) if dual_cost is not None and np.isfinite(dual_cost) else None
    upper = max(objective_value,upper) if upper is not None else None
    return dict(schedule=p,energy=s,revenue=revenue,
        terminal_value=float(terminal_value),objective_value=objective_value,
        objective_upper_bound=upper,
        certified_optimal=bool(result.success),solver_status=int(result.status),
        solver_message=str(result.message),
        revenue_upper_bound=upper if terminal_value==0 else None,
        absolute_bound_gap=None if upper is None else upper-objective_value,
        mip_gap=None if eta==1 else float(getattr(result,"mip_gap",np.nan)))


def daily_schedule(prices, initial_energy, eta, terminal_value=0.0):
    result = solve(prices,initial_energy,eta,time_limit=5,terminal_value=terminal_value)
    # Numerical clipping for the production daily scheduler only. The evaluator
    # does NOT repair invalid submitted outputs. The oracle is never repaired.
    energy = float(initial_energy)
    output = []
    for value in result["schedule"].ravel():
        p = float(np.clip(value,max(-2.5,-eta*energy),min(2.5,(5-energy)/eta)))
        output.append(p)
        energy = float(np.clip(energy+eta*max(p,0)+min(p,0)/eta,0,5))
    return np.asarray(output).reshape(24,1)
