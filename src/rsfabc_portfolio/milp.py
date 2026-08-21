"""Dinkelbach mixed-integer comparator for empirical maximum Omega."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix

from .baselines import equal_weight_k
from .objective import OmegaObjective
from .portfolio import repair_cardinality
from .types import OptimizationResult


@dataclass(frozen=True)
class OmegaMILPConfig:
    cardinality: int
    lower: float
    upper: float
    time_limit_seconds: float = 30.0
    mip_rel_gap: float = 1e-4
    residual_tolerance: float = 1e-7
    max_iterations: int = 30


@dataclass(frozen=True)
class _SignedOmegaOutcome:
    weights: np.ndarray
    objective: float
    success: bool
    message: str
    mip_gap: float | None
    q: float
    solver_calls: int
    convergence_best: list[float]


def _finite_float_or_none(value: float | None) -> float | None:
    """Return finite solver diagnostics without emitting invalid JSON numbers."""
    if value is None:
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _solve_signed_omega_milp(
    objective: OmegaObjective,
    config: OmegaMILPConfig,
    initial_weights: np.ndarray,
    deadline: float,
) -> _SignedOmegaOutcome:
    """Solve Omega directly when the optimum can lie below one.

    The fast formulation in :func:`solve_omega_milp` optimizes the monotone
    transform Omega-1 and is efficient while that ratio is non-negative.  A
    negative ratio makes its downside epigraph unbounded.  This formulation
    adds one sign binary per return scenario so gain and loss are exact, which
    keeps Dinkelbach subproblems bounded for every non-negative Omega value.
    """
    returns = objective.returns
    scenarios, n_assets = returns.shape
    k = config.cardinality
    alpha = objective.cost_rate / (2.0 * objective.amortization_days)

    w_slice = slice(0, n_assets)
    z_slice = slice(n_assets, 2 * n_assets)
    a_slice = slice(2 * n_assets, 3 * n_assets)
    g_slice = slice(3 * n_assets, 3 * n_assets + scenarios)
    d_slice = slice(3 * n_assets + scenarios, 3 * n_assets + 2 * scenarios)
    s_slice = slice(3 * n_assets + 2 * scenarios, 3 * n_assets + 3 * scenarios)
    variable_count = 3 * n_assets + 3 * scenarios

    row_count = 2 + 4 * n_assets + 3 * scenarios
    matrix = lil_matrix((row_count, variable_count), dtype=float)
    lower_rows = np.full(row_count, -np.inf)
    upper_rows = np.full(row_count, np.inf)
    row = 0

    matrix[row, w_slice] = 1.0
    lower_rows[row] = upper_rows[row] = 1.0
    row += 1
    matrix[row, z_slice] = 1.0
    lower_rows[row] = upper_rows[row] = float(k)
    row += 1

    for asset in range(n_assets):
        matrix[row, asset] = 1.0
        matrix[row, n_assets + asset] = -config.upper
        upper_rows[row] = 0.0
        row += 1
        matrix[row, asset] = -1.0
        matrix[row, n_assets + asset] = config.lower
        upper_rows[row] = 0.0
        row += 1
        matrix[row, asset] = 1.0
        matrix[row, 2 * n_assets + asset] = -1.0
        upper_rows[row] = float(objective.pretrade[asset])
        row += 1
        matrix[row, asset] = -1.0
        matrix[row, 2 * n_assets + asset] = -1.0
        upper_rows[row] = -float(objective.pretrade[asset])
        row += 1

    # A fully invested long-only portfolio has one-way turnover at most one.
    max_cost_shift = objective.cost_rate / objective.amortization_days
    positive_bounds = np.maximum(
        returns.max(axis=1) - objective.threshold, 0.0
    )
    negative_bounds = np.maximum(
        objective.threshold + max_cost_shift - returns.min(axis=1), 0.0
    )
    equality_constant = objective.threshold + alpha * objective.cash_pre
    for scenario in range(scenarios):
        gain_index = g_slice.start + scenario
        loss_index = d_slice.start + scenario
        sign_index = s_slice.start + scenario

        # r_t'w - alpha*sum(a) - tau - alpha*cash = gain_t - loss_t.
        matrix[row, w_slice] = returns[scenario]
        matrix[row, a_slice] = -alpha
        matrix[row, gain_index] = -1.0
        matrix[row, loss_index] = 1.0
        lower_rows[row] = upper_rows[row] = equality_constant
        row += 1

        # gain_t <= M+_t sign_t
        matrix[row, gain_index] = 1.0
        matrix[row, sign_index] = -positive_bounds[scenario]
        upper_rows[row] = 0.0
        row += 1

        # loss_t <= M-_t (1-sign_t)
        matrix[row, loss_index] = 1.0
        matrix[row, sign_index] = negative_bounds[scenario]
        upper_rows[row] = negative_bounds[scenario]
        row += 1

    if row != row_count:
        raise AssertionError("signed MILP constraint count mismatch")

    variable_lower = np.zeros(variable_count)
    variable_upper = np.full(variable_count, np.inf)
    variable_upper[w_slice] = config.upper
    variable_upper[z_slice] = 1.0
    variable_upper[a_slice] = 1.0
    variable_upper[g_slice] = positive_bounds
    variable_upper[d_slice] = negative_bounds
    variable_upper[s_slice] = 1.0
    bounds = Bounds(variable_lower, variable_upper)
    integrality = np.zeros(variable_count, dtype=int)
    integrality[z_slice] = 1
    integrality[s_slice] = 1
    constraints = LinearConstraint(csr_matrix(matrix), lower_rows, upper_rows)

    best_weights = repair_cardinality(
        initial_weights, k, config.lower, config.upper
    )
    best_stats = objective.evaluate(best_weights)
    best_score = best_stats.omega
    q = best_score
    best_gap: float | None = None
    solver_calls = 0
    success = False
    status = "signed scenario formulation did not converge"
    convergence_best = [best_score]

    for _ in range(config.max_iterations):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            status = "signed scenario formulation reached total time limit"
            break
        coefficients = np.zeros(variable_count)
        coefficients[g_slice] = -1.0 / scenarios
        coefficients[d_slice] = q / scenarios
        result = milp(
            c=coefficients,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={
                "time_limit": float(remaining),
                "mip_rel_gap": config.mip_rel_gap,
                "presolve": True,
                "disp": False,
            },
        )
        solver_calls += 1
        status = str(result.message)
        best_gap = getattr(result, "mip_gap", None)
        if result.x is None:
            break
        weights = repair_cardinality(
            np.asarray(result.x[w_slice], dtype=float),
            k,
            config.lower,
            config.upper,
        )
        stats = objective.evaluate(weights)
        if stats.omega > best_score:
            best_score = stats.omega
            best_weights = weights.copy()
        convergence_best.append(best_score)

        denominator = stats.loss + objective.epsilon
        q_new = stats.gain / denominator
        residual = stats.gain - q * denominator
        q = q_new
        if abs(residual) <= config.residual_tolerance and result.success:
            success = True
            status = f"signed scenario formulation converged; {result.message}"
            break
        if not result.success:
            status = f"signed scenario formulation not proven optimal; {result.message}"
            break

    return _SignedOmegaOutcome(
        weights=best_weights,
        objective=best_score,
        success=success,
        message=status,
        mip_gap=_finite_float_or_none(best_gap),
        q=float(q),
        solver_calls=solver_calls,
        convergence_best=convergence_best,
    )


def solve_omega_milp(
    objective: OmegaObjective,
    config: OmegaMILPConfig,
    fallback_weights: np.ndarray | None = None,
) -> OptimizationResult:
    start = time.perf_counter()
    deadline = start + config.time_limit_seconds
    returns = objective.returns
    scenarios, n_assets = returns.shape
    k = config.cardinality
    if k * config.lower > 1.0 + 1e-12 or k * config.upper < 1.0 - 1e-12:
        raise ValueError("infeasible cardinality and bounds")

    w_slice = slice(0, n_assets)
    z_slice = slice(n_assets, 2 * n_assets)
    a_slice = slice(2 * n_assets, 3 * n_assets)
    d_slice = slice(3 * n_assets, 3 * n_assets + scenarios)
    variable_count = 3 * n_assets + scenarios
    alpha = objective.cost_rate / (2.0 * objective.amortization_days)

    row_count = 2 + 4 * n_assets + scenarios
    matrix = lil_matrix((row_count, variable_count), dtype=float)
    lower_rows = np.full(row_count, -np.inf)
    upper_rows = np.full(row_count, np.inf)
    row = 0

    matrix[row, w_slice] = 1.0
    lower_rows[row] = upper_rows[row] = 1.0
    row += 1
    matrix[row, z_slice] = 1.0
    lower_rows[row] = upper_rows[row] = float(k)
    row += 1

    for asset in range(n_assets):
        matrix[row, asset] = 1.0
        matrix[row, n_assets + asset] = -config.upper
        upper_rows[row] = 0.0
        row += 1
        matrix[row, asset] = -1.0
        matrix[row, n_assets + asset] = config.lower
        upper_rows[row] = 0.0
        row += 1
        matrix[row, asset] = 1.0
        matrix[row, 2 * n_assets + asset] = -1.0
        upper_rows[row] = float(objective.pretrade[asset])
        row += 1
        matrix[row, asset] = -1.0
        matrix[row, 2 * n_assets + asset] = -1.0
        upper_rows[row] = -float(objective.pretrade[asset])
        row += 1

    downside_constant = -objective.threshold - alpha * objective.cash_pre
    for scenario in range(scenarios):
        matrix[row, w_slice] = -returns[scenario]
        matrix[row, a_slice] = alpha
        matrix[row, 3 * n_assets + scenario] = -1.0
        upper_rows[row] = downside_constant
        row += 1
    if row != row_count:
        raise AssertionError("MILP constraint count mismatch")

    variable_lower = np.zeros(variable_count)
    variable_upper = np.full(variable_count, np.inf)
    variable_upper[w_slice] = config.upper
    variable_upper[z_slice] = 1.0
    variable_upper[a_slice] = 1.0
    bounds = Bounds(variable_lower, variable_upper)
    integrality = np.zeros(variable_count, dtype=int)
    integrality[z_slice] = 1
    constraints = LinearConstraint(csr_matrix(matrix), lower_rows, upper_rows)

    q = 0.0
    best_weights: np.ndarray | None = None
    best_score = -np.inf
    best_status = ""
    best_gap: float | None = None
    convergence_best: list[float] = []
    convergence_iterations: list[int] = []
    solver_calls = 0
    success = False

    for iteration in range(config.max_iterations):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            best_status = "total time limit reached"
            break
        coefficients = np.zeros(variable_count)
        coefficients[w_slice] = -returns.mean(axis=0)
        coefficients[a_slice] = alpha
        coefficients[d_slice] = max(q, 0.0) / scenarios
        result = milp(
            c=coefficients,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={
                "time_limit": float(remaining),
                "mip_rel_gap": config.mip_rel_gap,
                "presolve": True,
                "disp": False,
            },
        )
        solver_calls += 1
        best_status = str(result.message)
        best_gap = getattr(result, "mip_gap", None)
        if result.x is None:
            break
        weights = np.asarray(result.x[w_slice], dtype=float)
        weights = repair_cardinality(weights, k, config.lower, config.upper)
        stats = objective.evaluate(weights)
        if stats.omega > best_score:
            best_score = stats.omega
            best_weights = weights.copy()
        convergence_iterations.append(iteration + 1)
        convergence_best.append(best_score)
        downside = stats.loss
        if downside <= objective.epsilon:
            success = True
            break
        # Omega - 1 = (mean excess - epsilon) / (loss + epsilon).
        denominator = downside + objective.epsilon
        q_new = stats.omega - 1.0
        residual = (stats.mean_excess - objective.epsilon) - q * denominator
        if q_new < -1e-10:
            signed = _solve_signed_omega_milp(
                objective,
                config,
                best_weights,
                deadline,
            )
            signed_stats = objective.evaluate(signed.weights)
            return OptimizationResult(
                method="omega_milp",
                weights=signed.weights,
                objective=signed_stats.omega,
                evaluations=0,
                runtime_seconds=time.perf_counter() - start,
                convergence_evaluations=list(
                    range(1, len(signed.convergence_best) + 1)
                ),
                convergence_best=signed.convergence_best,
                success=signed.success,
                message=(
                    "fast Omega-minus-one formulation detected Omega < 1; "
                    + signed.message
                ),
                diagnostics={
                    "solver_calls": solver_calls + signed.solver_calls,
                    "fast_solver_calls": solver_calls,
                    "signed_solver_calls": signed.solver_calls,
                    "signed_fallback_used": True,
                    "mip_gap": signed.mip_gap,
                    "dinkelbach_q": signed.q,
                    "turnover": signed_stats.turnover,
                },
            )
        if abs(residual) <= config.residual_tolerance:
            q = max(q_new, 0.0)
            success = bool(result.success)
            break
        q = max(q_new, 0.0)
        success = bool(result.success)

    if best_weights is None:
        if fallback_weights is not None:
            best_weights = repair_cardinality(
                fallback_weights, k, config.lower, config.upper
            )
            fallback_name = "prior portfolio"
        else:
            baseline = equal_weight_k(objective, k, config.lower, config.upper)
            best_weights = baseline.weights
            fallback_name = "EW-K"
        best_score = objective.score(best_weights)
        best_status = f"no feasible MILP incumbent; fallback={fallback_name}; {best_status}"
        success = False

    stats = objective.evaluate(best_weights)
    return OptimizationResult(
        method="omega_milp",
        weights=best_weights,
        objective=stats.omega,
        evaluations=0,
        runtime_seconds=time.perf_counter() - start,
        convergence_evaluations=convergence_iterations or [0],
        convergence_best=convergence_best or [stats.omega],
        success=success,
        message=best_status,
        diagnostics={
            "solver_calls": solver_calls,
            "fast_solver_calls": solver_calls,
            "signed_solver_calls": 0,
            "signed_fallback_used": False,
            "mip_gap": _finite_float_or_none(best_gap),
            "dinkelbach_q": float(q),
            "turnover": stats.turnover,
        },
    )
