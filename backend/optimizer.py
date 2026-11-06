"""MILP dispatch with ordered objectives instead of configurable weights."""

import atexit
from threading import Lock
from time import monotonic

import gurobipy as gp
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack

from backend.data import DT
from backend.schemas import CreateSimulation, OptimizationError

_solver_lock = Lock()
_environment = None


def close_solver():
    """Release the process's license after all models have finished."""
    global _environment
    with _solver_lock:
        if _environment is not None:
            _environment.dispose()
            _environment = None


atexit.register(close_solver)


def solve_dispatch(forecast, request: CreateSimulation, initial_soc):
    global _environment
    e, b = request.enterprise, request.equipment
    n = len(forecast)
    if not n:
        raise ValueError("The optimization window cannot be empty.")
    # Contiguous blocks keep the matrix readable and the number of variables linear.
    (
        buy,
        sell,
        charge,
        discharge,
        energy,
        curtail,
        excess,
        battery_on,
        grid_on,
    ) = (np.arange(i * n, (i + 1) * n) for i in range(9))
    peak = 9 * n
    size = peak + 1
    lower, upper, objective = np.zeros(size), np.full(size, np.inf), np.zeros(size)
    integrality = np.zeros(size)
    import_limit = b.grid_import_capacity_kw
    if e.import_hard_limit_kw is not None:
        import_limit = min(import_limit, e.import_hard_limit_kw)
    upper[buy], upper[sell] = import_limit, b.grid_export_capacity_kw
    upper[charge], upper[discharge] = b.charge_limit_kw, b.discharge_limit_kw
    lower[energy], upper[energy] = (
        e.reserve_soc * b.capacity_kwh,
        b.max_soc * b.capacity_kwh,
    )
    lower[energy[-1]] = b.terminal_soc * b.capacity_kwh
    upper[curtail] = [p["pv_kw"] for p in forecast]
    upper[battery_on], upper[grid_on] = 1, 1
    integrality[battery_on], integrality[grid_on] = 1, 1
    upper[peak] = import_limit
    objective[buy] = [p["buy_price"] * DT for p in forecast]
    objective[sell] = [-p["sell_price"] * DT for p in forecast]
    objective[charge] = objective[discharge] = 0.005 * DT
    if e.import_target_kw is None:
        upper[excess] = 0
    rows, lo, hi = [], [], []

    def add(coefficients, minimum=-np.inf, maximum=np.inf):
        rows.append(coefficients)
        lo.append(minimum)
        hi.append(maximum)

    for t, point in enumerate(forecast):
        net = point["load_kw"] - point["pv_kw"]
        add(
            {buy[t]: 1, sell[t]: -1, discharge[t]: 1, charge[t]: -1, curtail[t]: -1},
            net,
            net,
        )
        coefficients = {
            energy[t]: 1,
            charge[t]: -DT * b.efficiency,
            discharge[t]: DT / b.efficiency,
        }
        rhs = initial_soc * b.capacity_kwh if t == 0 else 0
        if t:
            coefficients[energy[t - 1]] = -1
        add(coefficients, rhs, rhs)
        add({charge[t]: 1, battery_on[t]: -b.charge_limit_kw}, maximum=0)
        add(
            {discharge[t]: 1, battery_on[t]: b.discharge_limit_kw},
            maximum=b.discharge_limit_kw,
        )
        add({buy[t]: 1, grid_on[t]: -import_limit}, maximum=0)
        add(
            {sell[t]: 1, grid_on[t]: b.grid_export_capacity_kw},
            maximum=b.grid_export_capacity_kw,
        )
        add({buy[t]: 1, peak: -1}, maximum=0)
        if e.import_target_kw is not None:
            add({buy[t]: 1, excess[t]: -1}, maximum=e.import_target_kw)

    matrix = lil_matrix((len(rows), size))
    for i, row in enumerate(rows):
        for j, value in row.items():
            matrix[i, j] = value
    matrix = matrix.tocsr()
    priorities = []
    if e.import_target_kw is not None:
        soft_objective = np.zeros(size)
        soft_objective[excess] = 1
        priorities.append(soft_objective)
    if e.objective == "peak":
        peak_objective = np.zeros(size)
        peak_objective[peak] = 1
        priorities.append(peak_objective)
    priorities.append(objective)
    # One environment avoids repeated WLS sessions. Its entire use is serialized
    # because API requests and automatic steps run on different Python threads.
    try:
        with _solver_lock:
            if _environment is None:
                environment = gp.Env(empty=True)
                try:
                    environment.setParam("OutputFlag", 0)
                    environment.start()
                except gp.GurobiError:
                    environment.dispose()
                    raise
                _environment = environment
            with gp.Model("ems_dispatch", env=_environment) as model:
                model.Params.Threads = 1
                model.Params.MIPGap = 0
                model.Params.MIPGapAbs = 0
                model.Params.DualReductions = 0
                model.Params.FeasibilityTol = 1e-8
                model.Params.IntFeasTol = 1e-8
                variables = model.addMVar(
                    size,
                    lb=lower,
                    ub=upper,
                    vtype=np.where(integrality, gp.GRB.BINARY, gp.GRB.CONTINUOUS),
                )
                lo_array, hi_array = np.asarray(lo), np.asarray(hi)
                equal = lo_array == hi_array
                for mask, sense, rhs in (
                    (equal, "=", lo_array),
                    (~equal & np.isfinite(lo_array), ">", lo_array),
                    (~equal & np.isfinite(hi_array), "<", hi_array),
                ):
                    if mask.any():
                        model.addMConstr(matrix[mask], variables, sense, rhs[mask])
                deadline = monotonic() + 10
                for stage, stage_objective in enumerate(priorities):
                    model.setObjective(stage_objective @ variables, gp.GRB.MINIMIZE)
                    model.Params.TimeLimit = max(0.0, deadline - monotonic())
                    model.optimize()
                    status = model.Status
                    if status != gp.GRB.OPTIMAL:
                        break
                    objective_value = float(model.ObjVal)
                    if stage < len(priorities) - 1:
                        # Keep the same objective priorities and numerical allowance.
                        bound = objective_value + 1e-6
                        model.addConstr(stage_objective @ variables <= bound)
                        matrix = vstack([matrix, csr_matrix(stage_objective)]).tocsr()
                        lo.append(-np.inf)
                        hi.append(bound)
                if status == gp.GRB.OPTIMAL:
                    x = variables.X.copy()
    except gp.GurobiError as error:
        raise OptimizationError(
            "Gurobi is unavailable. This step was not executed.",
            [
                "Check gurobi.lic, GRB_LICENSE_FILE and the connection to the license service.",
                f"Gurobi error code: {error.errno}",
            ],
            "solver_failed",
        ) from error
    if status != gp.GRB.OPTIMAL:
        reasons = []
        if status == gp.GRB.INFEASIBLE:
            for point in forecast:
                gap = (
                    point["load_kw"]
                    - point["pv_kw"]
                    - import_limit
                    - b.discharge_limit_kw
                )
                if gap > 1e-6:
                    reasons.append(
                        f"{point['time']}: supply is short by {gap:.3f} kW even at maximum battery discharge."
                    )
                    break
            if not reasons:
                reasons.append(
                    "Grid import limits, available energy, battery power or final SOC may conflict. This is not a minimal conflict diagnosis."
                )
            reasons.append(
                "Check the grid import hard limit, or use a soft target if appropriate. Hard constraints are never relaxed automatically."
            )
            raise OptimizationError(
                "Operating requirements conflict; no feasible dispatch plan.", reasons
            )
        raise OptimizationError(
            "Optimization did not complete successfully. This step was not executed.",
            [
                f"Gurobi solver status: {status}; the solver time limit may have been reached."
            ],
            "solver_failed",
        )
    # Verify physical feasibility before any action can be committed.
    ax = matrix.tocsr() @ x
    if (
        not np.isfinite(x).all()
        or np.any(ax < np.asarray(lo) - 1e-5)
        or np.any(ax > np.asarray(hi) + 1e-5)
        or np.any(x < lower - 1e-5)
        or np.any(x > upper + 1e-5)
        or np.any(np.abs(x[integrality == 1] - np.rint(x[integrality == 1])) > 1e-5)
    ):
        raise OptimizationError(
            "The solution failed physical validation. This step was not executed.",
            [],
            "solver_failed",
        )
    points = []
    for t, point in enumerate(forecast):
        imp, exp = max(0.0, float(x[buy[t]])), max(0.0, float(x[sell[t]]))
        points.append(
            {
                **point,
                "charge_kw": max(0.0, float(x[charge[t]])),
                "discharge_kw": max(0.0, float(x[discharge[t]])),
                "import_kw": imp,
                "export_kw": exp,
                "curtail_kw": max(0.0, float(x[curtail[t]])),
                "soc": float(x[energy[t]] / b.capacity_kwh),
                "cost_yuan": DT
                * (imp * point["buy_price"] - exp * point["sell_price"]),
                "target_excess_kw": max(0.0, imp - e.import_target_kw)
                if e.import_target_kw is not None
                else 0.0,
            }
        )
    return {
        "from_step": forecast[0]["slot"],
        "solver_status": "optimal",
        "objective_value": objective_value,
        "points": points,
    }
