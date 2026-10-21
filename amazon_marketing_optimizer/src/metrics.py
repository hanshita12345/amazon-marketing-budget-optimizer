"""Marketing KPI calculation engine for Amazon marketing campaigns.

Implements standard digital marketing and financial contribution metrics with strict
zero-division safety, handling of missing data, and clear distinction between ROAS and ROI.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def safe_divide(
    numerator: Union[float, int, pd.Series, np.ndarray],
    denominator: Union[float, int, pd.Series, np.ndarray],
    multiplier: float = 1.0,
    default: Optional[float] = None,
) -> Any:
    """Safely divide numerator by denominator handling zeros, nulls, and negative denominators.

    If denominator is 0 or NaN, returns default (typically None or NaN). Never returns inf or throws ZeroDivisionError.
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        num = pd.to_numeric(numerator, errors="coerce")
        den = pd.to_numeric(denominator, errors="coerce")
        # Mask where denominator is non-positive or null
        valid_mask = den.notna() & (den > 0) & num.notna()
        result = pd.Series(index=num.index, dtype=float)
        result[valid_mask] = (num[valid_mask] / den[valid_mask]) * multiplier
        result[~valid_mask] = np.nan if default is None else default
        return result

    if denominator is None or pd.isna(denominator) or denominator <= 0:
        return default
    if numerator is None or pd.isna(numerator):
        return default
    return (float(numerator) / float(denominator)) * multiplier


def calculate_click_through_rate(clicks: Any, impressions: Any) -> Any:
    """CTR (%) = (clicks / impressions) * 100"""
    return safe_divide(clicks, impressions, multiplier=100.0)


def calculate_conversion_rate(conversions: Any, clicks: Any) -> Any:
    """Conversion Rate (%) = (conversions / clicks) * 100"""
    return safe_divide(conversions, clicks, multiplier=100.0)


def calculate_cost_per_click(spend: Any, clicks: Any) -> Any:
    """CPC ($) = marketing_spend / clicks"""
    return safe_divide(spend, clicks)


def calculate_cost_per_acquisition(spend: Any, conversions: Any) -> Any:
    """CPA ($) = marketing_spend / conversions"""
    return safe_divide(spend, conversions)


def calculate_roas(revenue: Any, spend: Any) -> Any:
    """ROAS (x) = attributed_revenue / marketing_spend"""
    return safe_divide(revenue, spend)


def calculate_estimated_gross_profit(revenue: Any, margin_rate: Any) -> Any:
    """Estimated Gross Profit ($) = attributed_revenue * gross_margin_rate"""
    if isinstance(revenue, pd.Series) and isinstance(margin_rate, pd.Series):
        return (revenue * margin_rate).round(2)
    if revenue is None or margin_rate is None or pd.isna(revenue) or pd.isna(margin_rate):
        return None
    return round(float(revenue) * float(margin_rate), 2)


def calculate_marketing_contribution(gross_profit: Any, spend: Any) -> Any:
    """Estimated Marketing Contribution ($) = estimated_gross_profit - marketing_spend"""
    if isinstance(gross_profit, pd.Series) and isinstance(spend, pd.Series):
        return (gross_profit - spend).round(2)
    if gross_profit is None or spend is None or pd.isna(gross_profit) or pd.isna(spend):
        return None
    return round(float(gross_profit) - float(spend), 2)


def calculate_marketing_roi(gross_profit: Any, spend: Any) -> Any:
    """Marketing ROI (%) = ((estimated_gross_profit - marketing_spend) / marketing_spend) * 100.

    NOTE: Gross-profit-based ROI represents contribution margin before fixed overhead.
    It does not reflect total net company profit.
    """
    if isinstance(gross_profit, pd.Series) and isinstance(spend, pd.Series):
        contribution = gross_profit - spend
        return safe_divide(contribution, spend, multiplier=100.0)
    if gross_profit is None or spend is None or pd.isna(gross_profit) or pd.isna(spend):
        return None
    contribution = float(gross_profit) - float(spend)
    return safe_divide(contribution, spend, multiplier=100.0)


def calculate_cac(spend: Any, new_customers: Any) -> Any:
    """Customer Acquisition Cost ($) = marketing_spend / new_customers"""
    return safe_divide(spend, new_customers)


def calculate_arpc(revenue: Any, conversions: Any) -> Any:
    """Average Revenue Per Conversion (AOV) ($) = attributed_revenue / conversions"""
    return safe_divide(revenue, conversions)


def calculate_campaign_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate campaign-level records and calculate complete marketing KPI metrics."""
    if df.empty:
        return pd.DataFrame()

    agg_dict = {
        "channel": "first",
        "product_category": "first",
        "marketing_spend": "sum",
        "impressions": "sum",
        "clicks": "sum",
        "conversions": "sum",
        "attributed_revenue": "sum",
        "date": ["min", "max", "count"],
    }
    if "gross_margin_rate" in df.columns:
        # Weighted gross margin rate by revenue
        pass
    if "new_customers" in df.columns:
        agg_dict["new_customers"] = "sum"
    if "returning_customers" in df.columns:
        agg_dict["returning_customers"] = "sum"

    # Group by campaign
    grouped = df.groupby(["campaign_id", "campaign_name"]).agg({
        "channel": "first",
        "product_category": "first",
        "marketing_spend": "sum",
        "impressions": "sum",
        "clicks": "sum",
        "conversions": "sum",
        "attributed_revenue": "sum",
        "date": ["min", "max", "count"],
    }).reset_index()

    # Flatten multi-level columns
    grouped.columns = [
        "campaign_id",
        "campaign_name",
        "channel",
        "product_category",
        "marketing_spend",
        "impressions",
        "clicks",
        "conversions",
        "attributed_revenue",
        "start_date",
        "end_date",
        "active_days",
    ]

    # Calculate weighted gross margin rate per campaign
    if "gross_margin_rate" in df.columns:
        def calc_weighted_margin(sub_df: pd.DataFrame) -> float:
            rev_sum = sub_df["attributed_revenue"].sum()
            if rev_sum > 0:
                return float((sub_df["attributed_revenue"] * sub_df["gross_margin_rate"]).sum() / rev_sum)
            return float(sub_df["gross_margin_rate"].mean())

        margin_map = df.groupby("campaign_id").apply(calc_weighted_margin, include_groups=False).to_dict()
        grouped["gross_margin_rate"] = grouped["campaign_id"].map(margin_map).fillna(0.35)
    else:
        grouped["gross_margin_rate"] = 0.35

    # New customers sum if present
    if "new_customers" in df.columns and df["new_customers"].notna().any():
        new_cust_map = df.groupby("campaign_id")["new_customers"].sum().to_dict()
        grouped["new_customers"] = grouped["campaign_id"].map(new_cust_map)
    else:
        grouped["new_customers"] = np.nan

    # Calculate KPIs
    grouped["ctr"] = calculate_click_through_rate(grouped["clicks"], grouped["impressions"])
    grouped["cvr"] = calculate_conversion_rate(grouped["conversions"], grouped["clicks"])
    grouped["cpc"] = calculate_cost_per_click(grouped["marketing_spend"], grouped["clicks"])
    grouped["cpa"] = calculate_cost_per_acquisition(grouped["marketing_spend"], grouped["conversions"])
    grouped["roas"] = calculate_roas(grouped["attributed_revenue"], grouped["marketing_spend"])
    grouped["estimated_gross_profit"] = calculate_estimated_gross_profit(grouped["attributed_revenue"], grouped["gross_margin_rate"])
    grouped["marketing_contribution"] = calculate_marketing_contribution(grouped["estimated_gross_profit"], grouped["marketing_spend"])
    grouped["roi"] = calculate_marketing_roi(grouped["estimated_gross_profit"], grouped["marketing_spend"])
    grouped["cac"] = calculate_cac(grouped["marketing_spend"], grouped["new_customers"])
    grouped["arpc"] = calculate_arpc(grouped["attributed_revenue"], grouped["conversions"])

    return grouped


def calculate_channel_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate channel-level records and calculate marketing KPIs."""
    if df.empty:
        return pd.DataFrame()

    def agg_channel(g: pd.DataFrame) -> pd.Series:
        spend = g["marketing_spend"].sum()
        imprs = g["impressions"].sum()
        clicks = g["clicks"].sum()
        convs = g["conversions"].sum()
        rev = g["attributed_revenue"].sum()

        if "gross_margin_rate" in g.columns:
            if rev > 0:
                weighted_margin = (g["attributed_revenue"] * g["gross_margin_rate"]).sum() / rev
            else:
                weighted_margin = g["gross_margin_rate"].mean()
        else:
            weighted_margin = 0.35

        gross_profit = rev * weighted_margin
        contribution = gross_profit - spend

        new_cust = g["new_customers"].sum() if ("new_customers" in g.columns and g["new_customers"].notna().any()) else np.nan

        return pd.Series({
            "campaign_count": g["campaign_id"].nunique(),
            "marketing_spend": round(spend, 2),
            "impressions": int(imprs),
            "clicks": int(clicks),
            "conversions": int(convs),
            "attributed_revenue": round(rev, 2),
            "gross_margin_rate": round(weighted_margin, 3),
            "estimated_gross_profit": round(gross_profit, 2),
            "marketing_contribution": round(contribution, 2),
            "ctr": safe_divide(clicks, imprs, 100.0),
            "cvr": safe_divide(convs, clicks, 100.0),
            "cpc": safe_divide(spend, clicks),
            "cpa": safe_divide(spend, convs),
            "roas": safe_divide(rev, spend),
            "roi": safe_divide(contribution, spend, 100.0),
            "cac": safe_divide(spend, new_cust),
            "arpc": safe_divide(rev, convs),
        })

    chan_df = df.groupby("channel").apply(agg_channel, include_groups=False).reset_index()
    # Sort by spend descending
    chan_df = chan_df.sort_values(by="marketing_spend", ascending=False).reset_index(drop=True)
    return chan_df


def calculate_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall company-level marketing executive KPIs."""
    if df.empty:
        return {}

    total_spend = float(df["marketing_spend"].sum())
    total_imprs = int(df["impressions"].sum())
    total_clicks = int(df["clicks"].sum())
    total_convs = int(df["conversions"].sum())
    total_revenue = float(df["attributed_revenue"].sum())

    if "gross_margin_rate" in df.columns:
        if total_revenue > 0:
            avg_margin = float((df["attributed_revenue"] * df["gross_margin_rate"]).sum() / total_revenue)
        else:
            avg_margin = float(df["gross_margin_rate"].mean())
    else:
        avg_margin = 0.35

    estimated_gross_profit = round(total_revenue * avg_margin, 2)
    marketing_contribution = round(estimated_gross_profit - total_spend, 2)

    has_new_cust = "new_customers" in df.columns and df["new_customers"].notna().any()
    total_new_customers = int(df["new_customers"].sum()) if has_new_cust else None

    return {
        "total_spend": round(total_spend, 2),
        "total_impressions": total_imprs,
        "total_clicks": total_clicks,
        "total_conversions": total_convs,
        "total_revenue": round(total_revenue, 2),
        "overall_ctr": round(safe_divide(total_clicks, total_imprs, 100.0, default=0.0), 2),
        "overall_cvr": round(safe_divide(total_convs, total_clicks, 100.0, default=0.0), 2),
        "overall_cpc": round(safe_divide(total_spend, total_clicks, default=0.0), 2),
        "overall_cpa": round(safe_divide(total_spend, total_convs, default=0.0), 2),
        "overall_roas": round(safe_divide(total_revenue, total_spend, default=0.0), 2),
        "weighted_gross_margin": round(avg_margin, 3),
        "estimated_gross_profit": estimated_gross_profit,
        "marketing_contribution": marketing_contribution,
        "overall_roi": round(safe_divide(marketing_contribution, total_spend, 100.0, default=0.0), 2),
        "total_new_customers": total_new_customers,
        "overall_cac": round(safe_divide(total_spend, total_new_customers, default=0.0), 2) if total_new_customers else None,
        "overall_arpc": round(safe_divide(total_revenue, total_convs, default=0.0), 2),
    }


def format_currency(value: Optional[float], decimals: int = 2) -> str:
    """Format float as USD currency or 'N/A'."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_multiplier(value: Optional[float], decimals: int = 2) -> str:
    """Format float as multiplier (e.g. 3.45x) or 'N/A'."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}x"


def format_percentage(value: Optional[float], decimals: int = 2) -> str:
    """Format float as percentage (e.g. 12.50%) or 'N/A'."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"
