"""Constrained marketing budget optimization engine using SciPy and diminishing returns modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class ChannelConstraint:
    """Channel specific allocation constraints."""
    channel: str
    current_spend: float
    min_spend: float
    max_spend: float
    gross_margin_rate: float = 0.35
    historical_roas: float = 2.50
    historical_cpa: float = 25.00


@dataclass
class ChannelResponseCurve:
    """Parametric diminishing returns response curve for a marketing channel.

    Model: Revenue(spend) = alpha * ln(1 + beta * spend)
    Properties:
      - Revenue(0) = 0
      - Revenue(spend) is strictly increasing and concave (diminishing marginal returns)
      - Marginal ROAS at spend s is: dR/ds = (alpha * beta) / (1 + beta * spend)
    """
    channel: str
    alpha: float
    beta: float
    historical_spend: float
    historical_revenue: float
    gross_margin_rate: float

    def predict_revenue(self, spend: float) -> float:
        """Predict revenue at given spending level."""
        if spend <= 0:
            return 0.0
        return float(self.alpha * np.log(1.0 + self.beta * float(spend)))

    def predict_marginal_roas(self, spend: float) -> float:
        """Predict derivative dRevenue/dSpend at given spending level."""
        if spend < 0:
            return 0.0
        return float((self.alpha * self.beta) / (1.0 + self.beta * float(spend)))

    def predict_contribution(self, spend: float) -> float:
        """Predict gross profit minus marketing spend."""
        rev = self.predict_revenue(spend)
        return float(rev * self.gross_margin_rate - spend)


@dataclass
class OptimizationResult:
    """Comprehensive budget optimization results and comparison."""
    success: bool
    status_message: str
    total_budget: float
    allocatable_budget: float
    experiment_reserve: float
    reserve_pct: float
    current_total_spend: float
    current_expected_revenue: float
    current_expected_contribution: float
    recommended_total_spend: float
    recommended_expected_revenue: float
    recommended_expected_contribution: float
    incremental_revenue: float
    incremental_contribution: float
    overall_expected_roas: float
    channel_allocations: pd.DataFrame
    curves: Dict[str, ChannelResponseCurve]
    sensitivity_curve: pd.DataFrame
    is_fallback: bool = False
    infeasibility_notes: List[str] = field(default_factory=list)


def fit_channel_response_curve(
    channel: str,
    spend: float,
    revenue: float,
    margin_rate: float,
    efficiency_factor: float = 1.35,
) -> ChannelResponseCurve:
    """Fit a mathematically consistent diminishing returns curve calibrated to observed channel data.

    Solves:
      1) alpha * beta = initial_marginal_roas = efficiency_factor * observed_roas
      2) alpha * ln(1 + beta * spend) = revenue
    """
    spend = max(100.0, float(spend))
    revenue = max( spend * 0.5, float(revenue))
    observed_roas = revenue / spend
    r0 = observed_roas * efficiency_factor

    # Solve for beta such that (r0 / beta) * ln(1 + beta * spend) = revenue
    # (ln(1 + x) / x) = revenue / (r0 * spend) = observed_roas / r0 = 1 / efficiency_factor
    target_ratio = 1.0 / efficiency_factor

    # Numerical root search for x = beta * spend
    # Since efficiency_factor > 1, target_ratio < 1, unique root exists for x > 0
    low_x, high_x = 0.01, 30.0
    for _ in range(35):
        mid_x = (low_x + high_x) / 2.0
        val = np.log(1.0 + mid_x) / mid_x
        if val > target_ratio:
            low_x = mid_x
        else:
            high_x = mid_x

    x_sol = (low_x + high_x) / 2.0
    beta = x_sol / spend
    alpha = r0 / beta

    return ChannelResponseCurve(
        channel=channel,
        alpha=float(alpha),
        beta=float(beta),
        historical_spend=spend,
        historical_revenue=revenue,
        gross_margin_rate=float(margin_rate),
    )


def build_channel_curves_from_data(df: pd.DataFrame) -> Dict[str, ChannelResponseCurve]:
    """Fit response curves for all channels present in the campaign dataset."""
    curves = {}
    for ch, g in df.groupby("channel"):
        spend = float(g["marketing_spend"].sum())
        revenue = float(g["attributed_revenue"].sum())
        if "gross_margin_rate" in g.columns:
            if revenue > 0:
                margin = float((g["attributed_revenue"] * g["gross_margin_rate"]).sum() / revenue)
            else:
                margin = float(g["gross_margin_rate"].mean())
        else:
            margin = 0.35

        curves[ch] = fit_channel_response_curve(
            channel=ch,
            spend=spend,
            revenue=revenue,
            margin_rate=margin,
        )
    return curves


def optimize_budget(
    total_budget: float,
    channel_constraints: List[ChannelConstraint],
    curves: Dict[str, ChannelResponseCurve],
    experiment_reserve_pct: float = 0.05,
    min_target_roas: Optional[float] = None,
) -> OptimizationResult:
    """Perform constrained optimization of marketing budget across channels using SciPy SLSQP.

    Objective:
      Maximize Total Marketing Contribution = Sum( margin_c * Revenue_c(s_c) - s_c )
    Subject to:
      Sum(s_c) = Allocatable Budget = Total Budget * (1 - experiment_reserve_pct)
      min_spend_c <= s_c <= max_spend_c
    """
    channels = [cc.channel for cc in channel_constraints]
    num_channels = len(channels)
    infeasibility_notes: List[str] = []

    # 1. Budget splitting
    reserve_pct = float(np.clip(experiment_reserve_pct, 0.0, 0.30))
    experiment_reserve = round(total_budget * reserve_pct, 2)
    allocatable_budget = round(total_budget - experiment_reserve, 2)

    # 2. Extract bounds
    current_spends = np.array([float(cc.current_spend) for cc in channel_constraints])
    min_bounds = np.array([float(cc.min_spend) for cc in channel_constraints])
    max_bounds = np.array([float(cc.max_spend) for cc in channel_constraints])
    margins = np.array([float(cc.gross_margin_rate) for cc in channel_constraints])

    # Check bounds sanity
    for i, ch in enumerate(channels):
        if min_bounds[i] > max_bounds[i]:
            infeasibility_notes.append(
                f"Channel '{ch}' has min_spend (${min_bounds[i]:,.2f}) > max_spend (${max_bounds[i]:,.2f}). Adjusted max_spend."
            )
            max_bounds[i] = min_bounds[i] * 1.5

    sum_min = float(np.sum(min_bounds))
    sum_max = float(np.sum(max_bounds))

    # 3. Feasibility validation
    if sum_min > allocatable_budget:
        infeasibility_notes.append(
            f"Infeasible constraints: Sum of minimum channel budgets (${sum_min:,.2f}) "
            f"exceeds allocatable budget (${allocatable_budget:,.2f}). "
            "Minimum bounds have been proportionally scaled down to allow optimization."
        )
        scale_factor = (allocatable_budget * 0.98) / sum_min
        min_bounds = min_bounds * scale_factor

    if sum_max < allocatable_budget:
        infeasibility_notes.append(
            f"Budget constraint warning: Sum of maximum channel budgets (${sum_max:,.2f}) "
            f"is less than allocatable budget (${allocatable_budget:,.2f}). "
            "Maximum bounds expanded proportionally to utilize the total allocated budget."
        )
        expand_factor = (allocatable_budget * 1.05) / sum_max
        max_bounds = max_bounds * expand_factor

    bounds = [(float(min_bounds[i]), float(max_bounds[i])) for i in range(num_channels)]

    # Initial guess: proportional to current spend, clamped to bounds and normalized
    if np.sum(current_spends) > 0:
        x0 = (current_spends / np.sum(current_spends)) * allocatable_budget
    else:
        x0 = np.full(num_channels, allocatable_budget / num_channels)

    # Clamp and normalize x0
    x0 = np.clip(x0, min_bounds, max_bounds)
    x0 = (x0 / np.sum(x0)) * allocatable_budget

    # 4. Define Objective Function
    def objective(x: np.ndarray) -> float:
        total_contribution = 0.0
        for i, ch in enumerate(channels):
            curve = curves[ch]
            s = x[i]
            rev = curve.predict_revenue(s)
            contrib = rev * margins[i] - s
            total_contribution += contrib
        # Minimize negative contribution
        return -total_contribution

    # Jacobian for fast gradient descent
    def objective_grad(x: np.ndarray) -> np.ndarray:
        grad = np.zeros(num_channels)
        for i, ch in enumerate(channels):
            curve = curves[ch]
            s = x[i]
            drev_ds = curve.predict_marginal_roas(s)
            dcontrib_ds = margins[i] * drev_ds - 1.0
            grad[i] = -dcontrib_ds
        return grad

    # Budget equality constraint: sum(x) - allocatable_budget = 0
    budget_constraint = {
        "type": "eq",
        "fun": lambda x: np.sum(x) - allocatable_budget,
        "jac": lambda x: np.ones(num_channels),
    }

    # Optional soft target ROAS constraint
    constraints = [budget_constraint]
    if min_target_roas is not None and min_target_roas > 0:
        def roas_constraint(x: np.ndarray) -> float:
            tot_rev = sum(curves[ch].predict_revenue(x[i]) for i, ch in enumerate(channels))
            return (tot_rev / allocatable_budget) - min_target_roas
        constraints.append({"type": "ineq", "fun": roas_constraint})

    # 5. Run SciPy SLSQP Optimization
    opt_res = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 250, "ftol": 1e-6},
    )

    success = bool(opt_res.success)
    status_message = "Optimization completed successfully using SciPy SLSQP."

    if success:
        recommended_spends = np.maximum(min_bounds, opt_res.x)
    else:
        # Fallback allocation: distribute according to marginal contribution efficiency
        status_message = f"Optimizer warning: {opt_res.message}. Fallback marginal contribution allocation applied."
        infeasibility_notes.append(status_message)
        recommended_spends = _fallback_allocation(allocatable_budget, bounds, curves, channels, margins)

    # 6. Exact Budget Balancing (prevent rounding discrepancies)
    # Ensure exact sum equality down to the penny
    recommended_spends = np.clip(recommended_spends, min_bounds, max_bounds)
    recommended_spends = (recommended_spends / np.sum(recommended_spends)) * allocatable_budget
    recommended_spends = np.round(recommended_spends, 2)

    # Balance 1-cent rounding difference on largest allocation
    discrepancy = round(allocatable_budget - float(np.sum(recommended_spends)), 2)
    if abs(discrepancy) > 0:
        max_idx = int(np.argmax(recommended_spends))
        recommended_spends[max_idx] += discrepancy
        recommended_spends[max_idx] = round(recommended_spends[max_idx], 2)

    # 7. Compute Financial Comparison
    records = []
    curr_tot_rev, curr_tot_contrib = 0.0, 0.0
    rec_tot_rev, rec_tot_contrib = 0.0, 0.0

    for i, ch in enumerate(channels):
        curve = curves[ch]
        cc = channel_constraints[i]
        c_curr = float(cc.current_spend)
        c_rec = float(recommended_spends[i])

        curr_rev = curve.predict_revenue(c_curr)
        curr_contrib = curr_rev * margins[i] - c_curr
        curr_tot_rev += curr_rev
        curr_tot_contrib += curr_contrib

        rec_rev = curve.predict_revenue(c_rec)
        rec_contrib = rec_rev * margins[i] - c_rec
        rec_tot_rev += rec_rev
        rec_tot_contrib += rec_contrib

        budget_change = c_rec - c_curr
        budget_change_pct = (budget_change / c_curr * 100) if c_curr > 0 else 0.0
        exp_roas = (rec_rev / c_rec) if c_rec > 0 else 0.0
        cpa = cc.historical_cpa * (exp_roas / max(0.1, cc.historical_roas)) ** -0.5

        records.append({
            "channel": ch,
            "current_budget": round(c_curr, 2),
            "recommended_budget": round(c_rec, 2),
            "budget_change": round(budget_change, 2),
            "change_pct": round(budget_change_pct, 1),
            "current_expected_revenue": round(curr_rev, 2),
            "expected_revenue": round(rec_rev, 2),
            "revenue_change": round(rec_rev - curr_rev, 2),
            "current_expected_contribution": round(curr_contrib, 2),
            "expected_contribution": round(rec_contrib, 2),
            "contribution_change": round(rec_contrib - curr_contrib, 2),
            "expected_roas": round(exp_roas, 2),
            "expected_cpa": round(cpa, 2),
            "marginal_roas": round(curve.predict_marginal_roas(c_rec), 2),
        })

    alloc_df = pd.DataFrame(records)
    # Sort by recommended budget descending
    alloc_df = alloc_df.sort_values(by="recommended_budget", ascending=False).reset_index(drop=True)

    # 8. Compute Sensitivity Curve (Total budget from -30% to +30%)
    sens_records = []
    budget_multipliers = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
    for mult in budget_multipliers:
        sim_budget = round(total_budget * mult, 2)
        sim_allocatable = sim_budget * (1.0 - reserve_pct)
        # Scale recommendation to simulate sensitivity
        sim_spends = (recommended_spends / np.sum(recommended_spends)) * sim_allocatable
        sim_rev = sum(curves[ch].predict_revenue(sim_spends[i]) for i, ch in enumerate(channels))
        sim_contrib = sum(curves[ch].predict_revenue(sim_spends[i]) * margins[i] - sim_spends[i] for i, ch in enumerate(channels))
        sens_records.append({
            "budget_multiplier": f"{int(mult * 100)}%",
            "total_budget": round(sim_budget, 2),
            "expected_revenue": round(sim_rev, 2),
            "expected_contribution": round(sim_contrib, 2),
            "expected_roas": round(sim_rev / sim_allocatable, 2) if sim_allocatable > 0 else 0.0,
        })
    sens_df = pd.DataFrame(sens_records)

    return OptimizationResult(
        success=success,
        status_message=status_message,
        total_budget=round(total_budget, 2),
        allocatable_budget=round(allocatable_budget, 2),
        experiment_reserve=round(experiment_reserve, 2),
        reserve_pct=round(reserve_pct * 100, 1),
        current_total_spend=round(float(np.sum(current_spends)), 2),
        current_expected_revenue=round(curr_tot_rev, 2),
        current_expected_contribution=round(curr_tot_contrib, 2),
        recommended_total_spend=round(float(np.sum(recommended_spends)), 2),
        recommended_expected_revenue=round(rec_tot_rev, 2),
        recommended_expected_contribution=round(rec_tot_contrib, 2),
        incremental_revenue=round(rec_tot_rev - curr_tot_rev, 2),
        incremental_contribution=round(rec_tot_contrib - curr_tot_contrib, 2),
        overall_expected_roas=round(rec_tot_rev / allocatable_budget, 2) if allocatable_budget > 0 else 0.0,
        channel_allocations=alloc_df,
        curves=curves,
        sensitivity_curve=sens_df,
        is_fallback=not success,
        infeasibility_notes=infeasibility_notes,
    )


def _fallback_allocation(
    allocatable_budget: float,
    bounds: List[Tuple[float, float]],
    curves: Dict[str, ChannelResponseCurve],
    channels: List[str],
    margins: np.ndarray,
) -> np.ndarray:
    """Robust fallback: Allocates budget based on marginal efficiency weights when SLSQP fails."""
    num = len(channels)
    min_b = np.array([b[0] for b in bounds])
    max_b = np.array([b[1] for b in bounds])

    # Start at minimums
    spends = min_b.copy()
    remaining = allocatable_budget - np.sum(spends)

    if remaining <= 0:
        return (spends / np.sum(spends)) * allocatable_budget

    # Weights proportional to marginal return at current baseline
    weights = np.array([
        max(0.1, curves[ch].predict_marginal_roas(min_b[i]) * margins[i])
        for i, ch in enumerate(channels)
    ])
    weights = weights / np.sum(weights)

    spends += remaining * weights
    return np.clip(spends, min_b, max_b)
