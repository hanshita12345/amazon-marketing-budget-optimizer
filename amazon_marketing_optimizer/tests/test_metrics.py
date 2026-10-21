"""Unit tests for marketing KPI calculation engine and zero-division safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.metrics import (
    calculate_arpc,
    calculate_cac,
    calculate_campaign_kpis,
    calculate_channel_kpis,
    calculate_click_through_rate,
    calculate_conversion_rate,
    calculate_cost_per_acquisition,
    calculate_cost_per_click,
    calculate_estimated_gross_profit,
    calculate_executive_kpis,
    calculate_marketing_contribution,
    calculate_marketing_roi,
    calculate_roas,
    format_currency,
    format_multiplier,
    format_percentage,
    safe_divide,
)


def test_standard_kpi_formulas():
    """Verify mathematical correctness of standard marketing metrics."""
    spend = 500.0
    impressions = 20000
    clicks = 500
    conversions = 25
    revenue = 2500.0
    margin_rate = 0.40
    new_customers = 15

    # CTR = 500 / 20000 * 100 = 2.50%
    assert calculate_click_through_rate(clicks, impressions) == 2.50

    # CVR = 25 / 500 * 100 = 5.0%
    assert calculate_conversion_rate(conversions, clicks) == 5.0

    # CPC = 500 / 500 = $1.00
    assert calculate_cost_per_click(spend, clicks) == 1.00

    # CPA = 500 / 25 = $20.00
    assert calculate_cost_per_acquisition(spend, conversions) == 20.00

    # ROAS = 2500 / 500 = 5.0x
    assert calculate_roas(revenue, spend) == 5.0

    # Gross profit = 2500 * 0.40 = $1000.00
    gp = calculate_estimated_gross_profit(revenue, margin_rate)
    assert gp == 1000.00

    # Contribution = 1000 - 500 = $500.00
    contrib = calculate_marketing_contribution(gp, spend)
    assert contrib == 500.00

    # ROI = (1000 - 500) / 500 * 100 = 100.0%
    assert calculate_marketing_roi(gp, spend) == 100.00

    # CAC = 500 / 15 = $33.33
    cac = calculate_cac(spend, new_customers)
    assert abs(cac - 33.3333) < 0.01

    # ARPC = 2500 / 25 = $100.00
    assert calculate_arpc(revenue, conversions) == 100.00


def test_zero_spend_campaigns():
    """Verify that zero marketing spend is handled safely without division by zero errors."""
    zero_spend = 0.0
    clicks = 100
    conversions = 10
    revenue = 500.0

    # CPC with 0 spend is 0.0
    assert calculate_cost_per_click(zero_spend, clicks) == 0.0

    # CPA with 0 spend is 0.0
    assert calculate_cost_per_acquisition(zero_spend, conversions) == 0.0

    # ROAS with 0 spend (denominator = 0): safe_divide must return None / default, not throw exception or return inf
    roas = calculate_roas(revenue, zero_spend)
    assert roas is None

    # ROI with 0 spend
    roi = calculate_marketing_roi(200.0, zero_spend)
    assert roi is None


def test_zero_clicks_campaigns():
    """Verify safe calculation when a campaign has zero clicks."""
    impressions = 5000
    clicks = 0
    spend = 100.0

    # CTR with 0 clicks
    assert calculate_click_through_rate(clicks, impressions) == 0.0

    # CPC with 0 clicks (denominator = 0)
    assert calculate_cost_per_click(spend, clicks) is None

    # CVR with 0 clicks (denominator = 0)
    assert calculate_conversion_rate(5, clicks) is None


def test_zero_conversions_campaigns():
    """Verify safe calculation when conversions are zero."""
    clicks = 250
    conversions = 0
    spend = 150.0
    revenue = 0.0

    # CVR with 0 conversions
    assert calculate_conversion_rate(conversions, clicks) == 0.0

    # CPA with 0 conversions (denominator = 0)
    assert calculate_cost_per_acquisition(spend, conversions) is None

    # ARPC with 0 conversions
    assert calculate_arpc(revenue, conversions) is None


def test_missing_gross_margin_data():
    """Verify campaign aggregation safely defaults missing margin to 0.35 baseline."""
    df = pd.DataFrame({
        "date": ["2026-09-01"],
        "campaign_id": ["C1"],
        "campaign_name": ["Camp 1"],
        "channel": ["Google Ads"],
        "product_category": ["General"],
        "marketing_spend": [100.0],
        "impressions": [2000],
        "clicks": [50],
        "conversions": [5],
        "attributed_revenue": [300.0],
        # No gross_margin_rate column provided
    })

    kpis = calculate_campaign_kpis(df)
    assert len(kpis) == 1
    assert "gross_margin_rate" in kpis.columns
    assert kpis["gross_margin_rate"].iloc[0] == 0.35
    assert kpis["estimated_gross_profit"].iloc[0] == round(300.0 * 0.35, 2)


def test_missing_new_customers_data():
    """Verify that CAC safely returns NaN/None when new customer data is not present."""
    df = pd.DataFrame({
        "date": ["2026-09-01"],
        "campaign_id": ["C1"],
        "campaign_name": ["Camp 1"],
        "channel": ["Google Ads"],
        "product_category": ["General"],
        "marketing_spend": [100.0],
        "impressions": [2000],
        "clicks": [50],
        "conversions": [5],
        "attributed_revenue": [300.0],
        # No new_customers column
    })

    kpis = calculate_campaign_kpis(df)
    assert pd.isna(kpis["cac"].iloc[0])


def test_formatters_null_safety():
    """Verify formatters output 'N/A' for None or NaN without throwing exceptions."""
    assert format_currency(None) == "N/A"
    assert format_currency(np.nan) == "N/A"
    assert format_currency(1250.5) == "$1,250.50"

    assert format_multiplier(None) == "N/A"
    assert format_multiplier(np.nan) == "N/A"
    assert format_multiplier(3.1415) == "3.14x"

    assert format_percentage(None) == "N/A"
    assert format_percentage(np.nan) == "N/A"
    assert format_percentage(12.34) == "12.34%"
