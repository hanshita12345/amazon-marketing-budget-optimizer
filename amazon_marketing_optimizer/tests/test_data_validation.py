"""Unit and integration tests for data validation module."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data_loader import generate_synthetic_dataset
from src.data_validation import map_columns, validate_marketing_data


def test_valid_synthetic_dataset_validation():
    """Verify that generated synthetic dataset passes full validation."""
    df = generate_synthetic_dataset(num_days=30, seed=123)
    result = validate_marketing_data(df)

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.cleaned_df is not None
    assert len(result.cleaned_df) == len(df)
    assert result.summary["valid_rows"] == len(df)
    assert result.summary["total_spend"] > 0
    assert result.summary["total_revenue"] > 0


def test_missing_required_columns():
    """Verify that missing mandatory columns are detected and flagged as errors."""
    incomplete_df = pd.DataFrame({
        "date": ["2026-09-01", "2026-09-02"],
        "campaign_id": ["C1", "C2"],
        "campaign_name": ["Camp 1", "Camp 2"],
        "channel": ["Google Ads", "Google Ads"],
        # Missing marketing_spend, impressions, clicks, conversions, attributed_revenue
    })

    result = validate_marketing_data(incomplete_df)
    assert result.is_valid is False
    assert any("Missing required columns" in err for err in result.errors)
    assert result.cleaned_df is None


def test_column_alias_mapping():
    """Verify that alternative column headers are automatically mapped to canonical schema."""
    user_df = pd.DataFrame({
        "day": ["2026-09-01"],
        "cid": ["C1"],
        "cname": ["Test Campaign"],
        "source": ["Amazon Sponsored Products"],
        "spend": [150.0],
        "imprs": [5000],
        "clicks": [120],
        "orders": [15],
        "sales": [750.0],
        "margin": [0.40],
    })

    renamed_df, mapping_applied, missing = map_columns(user_df)
    assert "marketing_spend" in renamed_df.columns
    assert "conversions" in renamed_df.columns
    assert "attributed_revenue" in renamed_df.columns
    assert len(missing) == 0

    val_res = validate_marketing_data(user_df)
    assert val_res.is_valid is True
    assert val_res.cleaned_df is not None
    assert val_res.cleaned_df["marketing_spend"].iloc[0] == 150.0


def test_negative_values_rejection():
    """Verify that negative spending or revenue is rejected."""
    invalid_df = pd.DataFrame({
        "date": ["2026-09-01"],
        "campaign_id": ["C1"],
        "campaign_name": ["Camp 1"],
        "channel": ["Meta Ads"],
        "marketing_spend": [-50.0],  # Invalid negative spend
        "impressions": [1000],
        "clicks": [50],
        "conversions": [5],
        "attributed_revenue": [250.0],
    })

    result = validate_marketing_data(invalid_df)
    assert result.is_valid is False
    assert any("invalid negative values" in err for err in result.errors)


def test_logical_hierarchy_violations():
    """Verify that clicks > impressions or conversions > clicks triggers validation error."""
    # Test clicks > impressions
    bad_clicks_df = pd.DataFrame({
        "date": ["2026-09-01"],
        "campaign_id": ["C1"],
        "campaign_name": ["Camp 1"],
        "channel": ["Google Ads"],
        "marketing_spend": [100.0],
        "impressions": [50],
        "clicks": [100],  # Logical violation
        "conversions": [5],
        "attributed_revenue": [200.0],
    })
    res1 = validate_marketing_data(bad_clicks_df)
    assert res1.is_valid is False
    assert any("clicks exceed impressions" in err for err in res1.errors)

    # Test conversions > clicks
    bad_convs_df = pd.DataFrame({
        "date": ["2026-09-01"],
        "campaign_id": ["C1"],
        "campaign_name": ["Camp 1"],
        "channel": ["Google Ads"],
        "marketing_spend": [100.0],
        "impressions": [1000],
        "clicks": [10],
        "conversions": [25],  # Logical violation
        "attributed_revenue": [200.0],
    })
    res2 = validate_marketing_data(bad_convs_df)
    assert res2.is_valid is False
    assert any("conversions exceed clicks" in err for err in res2.errors)


def test_empty_dataframe_handling():
    """Verify safe handling when empty or None DataFrame is provided."""
    res_none = validate_marketing_data(None)
    assert res_none.is_valid is False
    assert len(res_none.errors) > 0

    res_empty = validate_marketing_data(pd.DataFrame())
    assert res_empty.is_valid is False


def test_gross_margin_percentage_normalization():
    """Verify that margin given as 0-100% (e.g. 35.0) is normalized to decimal (0.35)."""
    margin_df = pd.DataFrame({
        "date": ["2026-09-01"],
        "campaign_id": ["C1"],
        "campaign_name": ["Camp 1"],
        "channel": ["Email Marketing"],
        "marketing_spend": [25.0],
        "impressions": [1000],
        "clicks": [50],
        "conversions": [5],
        "attributed_revenue": [250.0],
        "gross_margin_rate": [35.0],  # Provided as percentage
    })

    result = validate_marketing_data(margin_df)
    assert result.is_valid is True
    assert result.cleaned_df is not None
    assert abs(result.cleaned_df["gross_margin_rate"].iloc[0] - 0.35) < 1e-4
    assert any("normalized from percentage" in w for w in result.warnings)


def test_duplicate_record_aggregation():
    """Verify that duplicate (date, campaign_id) entries are aggregated gracefully."""
    dup_df = pd.DataFrame({
        "date": ["2026-09-01", "2026-09-01"],
        "campaign_id": ["C1", "C1"],
        "campaign_name": ["Echo Promo", "Echo Promo"],
        "channel": ["Amazon Sponsored Products", "Amazon Sponsored Products"],
        "marketing_spend": [100.0, 50.0],
        "impressions": [2000, 1000],
        "clicks": [40, 20],
        "conversions": [4, 2],
        "attributed_revenue": [200.0, 100.0],
        "gross_margin_rate": [0.30, 0.30],
    })

    result = validate_marketing_data(dup_df)
    assert result.is_valid is True
    assert len(result.cleaned_df) == 1
    assert result.cleaned_df["marketing_spend"].iloc[0] == 150.0
    assert result.cleaned_df["attributed_revenue"].iloc[0] == 300.0
    assert any("duplicate records" in w for w in result.warnings)
