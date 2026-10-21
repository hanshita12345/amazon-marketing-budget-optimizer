"""Campaign performance classification and efficiency analysis engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from src.metrics import calculate_campaign_kpis, format_currency, format_multiplier


@dataclass
class PerformanceThresholds:
    """Configurable business thresholds for campaign classification."""
    min_roas: float = 2.50
    max_cpa: float = 35.00
    min_conversions: int = 10
    min_spend: float = 150.00
    min_impressions: int = 500
    min_clicks: int = 30


def classify_single_campaign(
    row: pd.Series,
    thresholds: Optional[PerformanceThresholds] = None,
) -> Dict[str, str]:
    """Classify a single campaign into one of five performance categories.

    Categories:
      1. High Performing
      2. Performing Normally
      3. Underperforming
      4. Potentially Inefficient
      5. Insufficient Data

    Returns dict with 'classification' and detailed 'reason'.
    """
    if thresholds is None:
        thresholds = PerformanceThresholds()

    spend = float(row.get("marketing_spend", 0.0) or 0.0)
    conversions = int(row.get("conversions", 0) or 0)
    impressions = int(row.get("impressions", 0) or 0)
    clicks = int(row.get("clicks", 0) or 0)
    roas = float(row.get("roas", 0.0) or 0.0) if pd.notna(row.get("roas")) else 0.0
    cpa = float(row.get("cpa", 0.0) or 0.0) if pd.notna(row.get("cpa")) else float("inf")
    contribution = float(row.get("marketing_contribution", 0.0) or 0.0) if pd.notna(row.get("marketing_contribution")) else 0.0

    # 1. Check for Insufficient Evidence first
    evidence_deficits = []
    if spend < thresholds.min_spend:
        evidence_deficits.append(f"spend ({format_currency(spend)}) < {format_currency(thresholds.min_spend)}")
    if conversions < thresholds.min_conversions:
        evidence_deficits.append(f"conversions ({conversions}) < {thresholds.min_conversions}")
    if impressions < thresholds.min_impressions:
        evidence_deficits.append(f"impressions ({impressions:,}) < {thresholds.min_impressions:,}")
    if clicks < thresholds.min_clicks:
        evidence_deficits.append(f"clicks ({clicks}) < {thresholds.min_clicks}")

    # If 2 or more evidence thresholds are not met, mark as Insufficient Data
    if len(evidence_deficits) >= 2 or conversions < 3 or spend < (thresholds.min_spend * 0.5):
        reason = (
            f"Campaign has insufficient sample size to evaluate reliably ({', '.join(evidence_deficits)}). "
            f"Observed ROAS: {format_multiplier(roas)}, Conversions: {conversions}."
        )
        return {
            "classification": "Insufficient Data",
            "reason": reason,
            "badge_color": "#6c757d",
        }

    # 2. High Performing
    # Exceeds ROAS target by 20%+ and CPA is below threshold, or contribution is strongly positive
    if (roas >= thresholds.min_roas * 1.20 and cpa <= thresholds.max_cpa) or (roas >= thresholds.min_roas * 1.50):
        reason = (
            f"ROAS of {format_multiplier(roas)} exceeds target ({format_multiplier(thresholds.min_roas)}) "
            f"with efficient CPA of {format_currency(cpa)} (target <= {format_currency(thresholds.max_cpa)}) "
            f"and positive marketing contribution of {format_currency(contribution)} across {conversions} conversions."
        )
        return {
            "classification": "High Performing",
            "reason": reason,
            "badge_color": "#28a745",
        }

    # 3. Performing Normally
    # Meets or near ROAS target and CPA is acceptable
    if roas >= thresholds.min_roas * 0.90 and cpa <= thresholds.max_cpa * 1.15 and contribution >= 0:
        reason = (
            f"ROAS ({format_multiplier(roas)}) and CPA ({format_currency(cpa)}) align within standard target range "
            f"(Target ROAS: {format_multiplier(thresholds.min_roas)}, Target CPA: {format_currency(thresholds.max_cpa)})."
        )
        return {
            "classification": "Performing Normally",
            "reason": reason,
            "badge_color": "#007bff",
        }

    # 4. Potentially Inefficient
    # Has sufficient data, but ROAS is severely low (< 1.5x) or CPA is inflated and contribution is negative
    if roas < (thresholds.min_roas * 0.70) or (cpa > thresholds.max_cpa * 1.30 and contribution < 0) or (contribution < -200 and roas < thresholds.min_roas):
        reason = (
            f"ROAS ({format_multiplier(roas)}) is significantly below target ({format_multiplier(thresholds.min_roas)}) "
            f"and CPA ({format_currency(cpa)}) exceeds threshold ({format_currency(thresholds.max_cpa)}), "
            f"generating negative marketing contribution of {format_currency(contribution)} over ${spend:,.2f} spend."
        )
        return {
            "classification": "Potentially Inefficient",
            "reason": reason,
            "badge_color": "#dc3545",
        }

    # 5. Underperforming
    # Does not meet targets, but not critical enough to be labeled completely inefficient
    reason = (
        f"Campaign performance ({format_multiplier(roas)} ROAS, {format_currency(cpa)} CPA) trails target benchmarks. "
        f"Requires bid or creative optimization."
    )
    return {
        "classification": "Underperforming",
        "reason": reason,
        "badge_color": "#ffc107",
    }


def analyze_campaign_performance(
    df: pd.DataFrame,
    thresholds: Optional[PerformanceThresholds] = None,
) -> pd.DataFrame:
    """Analyze campaign performance, compute KPIs, and classify each campaign."""
    if thresholds is None:
        thresholds = PerformanceThresholds()

    # Ensure campaign-level KPIs are computed
    if "roas" not in df.columns or "cpa" not in df.columns or "active_days" not in df.columns:
        kpi_df = calculate_campaign_kpis(df)
    else:
        kpi_df = df.copy()

    if kpi_df.empty:
        return pd.DataFrame()

    classifications = []
    reasons = []
    badge_colors = []

    for _, row in kpi_df.iterrows():
        res = classify_single_campaign(row, thresholds)
        classifications.append(res["classification"])
        reasons.append(res["reason"])
        badge_colors.append(res["badge_color"])

    kpi_df["classification"] = classifications
    kpi_df["classification_reason"] = reasons
    kpi_df["badge_color"] = badge_colors

    # Sort: Inefficient & Underperforming first or by spend descending
    kpi_df = kpi_df.sort_values(by="marketing_spend", ascending=False).reset_index(drop=True)
    return kpi_df


def get_performance_distribution(analyzed_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary counts and financial metrics by classification category."""
    if analyzed_df.empty or "classification" not in analyzed_df.columns:
        return {}

    dist = {}
    for cat in ["High Performing", "Performing Normally", "Underperforming", "Potentially Inefficient", "Insufficient Data"]:
        sub = analyzed_df[analyzed_df["classification"] == cat]
        dist[cat] = {
            "count": len(sub),
            "total_spend": float(sub["marketing_spend"].sum()) if not sub.empty else 0.0,
            "total_revenue": float(sub["attributed_revenue"].sum()) if not sub.empty else 0.0,
            "total_contribution": float(sub["marketing_contribution"].sum()) if not sub.empty else 0.0,
            "avg_roas": float(sub["attributed_revenue"].sum() / sub["marketing_spend"].sum()) if not sub.empty and sub["marketing_spend"].sum() > 0 else 0.0,
        }
    return dist
