"""Unit tests for SciPy constrained marketing budget optimizer."""

from __future__ import annotations

import numpy as np
import pytest
from src.budget_optimizer import (
    ChannelConstraint,
    ChannelResponseCurve,
    fit_channel_response_curve,
    optimize_budget,
)


def _get_mock_curves_and_constraints():
    """Helper creating standard multi-channel curves and constraints."""
    channels_info = [
        ("Amazon Sponsored Products", 20000.0, 80000.0, 0.35, 10000.0, 45000.0),
        ("Amazon Sponsored Brands", 15000.0, 45000.0, 0.35, 6000.0, 30000.0),
        ("Google Ads", 12000.0, 30000.0, 0.35, 5000.0, 25000.0),
        ("Meta Ads", 8000.0, 10000.0, 0.35, 3000.0, 18000.0),
        ("Email Marketing", 3000.0, 35000.0, 0.35, 1000.0, 10000.0),
    ]

    curves = {}
    constraints = []

    for name, spend, rev, margin, min_s, max_s in channels_info:
        curve = fit_channel_response_curve(
            channel=name,
            spend=spend,
            revenue=rev,
            margin_rate=margin,
            efficiency_factor=1.35,
        )
        curves[name] = curve
        constraints.append(
            ChannelConstraint(
                channel=name,
                current_spend=spend,
                min_spend=min_s,
                max_spend=max_s,
                gross_margin_rate=margin,
                historical_roas=rev / spend,
                historical_cpa=spend / 500,
            )
        )

    return curves, constraints


def test_budget_allocation_sums_to_allocatable_budget():
    """Verify that recommended channel allocations sum exactly to the allocatable budget."""
    curves, constraints = _get_mock_curves_and_constraints()
    total_budget = 60000.0
    reserve_pct = 0.05  # 5% reserve = $3,000, allocatable = $57,000

    result = optimize_budget(
        total_budget=total_budget,
        channel_constraints=constraints,
        curves=curves,
        experiment_reserve_pct=reserve_pct,
    )

    assert bool(result.success) is True
    assert result.allocatable_budget == 57000.0
    assert result.experiment_reserve == 3000.0

    rec_sum = round(float(result.channel_allocations["recommended_budget"].sum()), 2)
    assert rec_sum == 57000.0
    assert abs(rec_sum - result.allocatable_budget) < 1e-4


def test_channel_bounds_constraints_respected():
    """Verify that all channel recommended allocations respect their min and max bounds."""
    curves, constraints = _get_mock_curves_and_constraints()
    total_budget = 65000.0

    result = optimize_budget(
        total_budget=total_budget,
        channel_constraints=constraints,
        curves=curves,
        experiment_reserve_pct=0.05,
    )

    alloc_map = result.channel_allocations.set_index("channel")["recommended_budget"].to_dict()

    for cc in constraints:
        rec = alloc_map[cc.channel]
        # Allow slight 1-cent rounding tolerance
        assert rec >= (cc.min_spend - 0.05), f"Channel {cc.channel} spend {rec} below min {cc.min_spend}"
        assert rec <= (cc.max_spend + 0.05), f"Channel {cc.channel} spend {rec} above max {cc.max_spend}"


def test_diminishing_marginal_returns():
    """Verify that response curve exhibits concave diminishing marginal returns."""
    curve = fit_channel_response_curve(
        channel="Amazon Sponsored Products",
        spend=10000.0,
        revenue=40000.0,
        margin_rate=0.35,
    )

    # Revenue must be strictly increasing
    r_small = curve.predict_revenue(5000.0)
    r_med = curve.predict_revenue(10000.0)
    r_large = curve.predict_revenue(20000.0)
    assert r_small < r_med < r_large

    # Marginal ROAS must be strictly decreasing (concavity)
    m_small = curve.predict_marginal_roas(5000.0)
    m_med = curve.predict_marginal_roas(10000.0)
    m_large = curve.predict_marginal_roas(20000.0)
    assert m_small > m_med > m_large > 0


def test_infeasible_budget_constraints_handling():
    """Verify that infeasible constraints (sum of minimums > total budget) are handled gracefully."""
    curves, constraints = _get_mock_curves_and_constraints()
    # Set sum of minimum spends ($25,000) to exceed total allocatable budget ($15,000)
    total_small_budget = 15000.0

    result = optimize_budget(
        total_budget=total_small_budget,
        channel_constraints=constraints,
        curves=curves,
        experiment_reserve_pct=0.05,
    )

    # Should not crash, must note infeasibility, and allocations should sum to allocatable budget
    assert len(result.infeasibility_notes) > 0
    rec_sum = round(float(result.channel_allocations["recommended_budget"].sum()), 2)
    assert abs(rec_sum - result.allocatable_budget) < 1e-4


def test_sensitivity_curve_generated():
    """Verify that sensitivity curve table is generated across budget multipliers."""
    curves, constraints = _get_mock_curves_and_constraints()
    result = optimize_budget(
        total_budget=50000.0,
        channel_constraints=constraints,
        curves=curves,
    )

    assert not result.sensitivity_curve.empty
    assert "budget_multiplier" in result.sensitivity_curve.columns
    assert "expected_revenue" in result.sensitivity_curve.columns
    assert len(result.sensitivity_curve) >= 5
