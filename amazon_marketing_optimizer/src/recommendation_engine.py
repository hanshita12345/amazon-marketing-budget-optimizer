"""Explainable business recommendation engine for marketing budget and campaign optimization."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd
from src.budget_optimizer import OptimizationResult
from src.campaign_analysis import PerformanceThresholds
from src.metrics import format_currency, format_multiplier, format_percentage


@dataclass
class Recommendation:
    """Actionable business recommendation derived from empirical campaign analysis."""
    priority: str                       # "HIGH", "MEDIUM", "LOW"
    category: str                       # e.g. "Budget Reallocation", "Campaign Efficiency", "Conversion Rate", "Bidding Strategy", "Experimentation"
    target_name: str                    # Campaign or Channel Name
    target_type: str                    # "channel" or "campaign"
    evidence: str                       # Numerical proof from calculated metrics
    business_explanation: str          # Strategic business context
    recommended_action: str             # Concrete tactical next step
    estimated_budget_impact: str        # Financial budget impact description
    confidence_limitation: str          # Data sufficiency and modeling assumptions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_recommendations(
    campaign_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    opt_result: Optional[OptimizationResult] = None,
    thresholds: Optional[PerformanceThresholds] = None,
) -> List[Recommendation]:
    """Generate prioritized, data-driven marketing recommendations based on empirical performance."""
    if thresholds is None:
        thresholds = PerformanceThresholds()

    recommendations: List[Recommendation] = []

    # 1. Budget Reallocation Recommendations from Optimizer Results
    if opt_result is not None and not opt_result.channel_allocations.empty:
        alloc_df = opt_result.channel_allocations

        # Channels recommended for significant expansion (> $500 increase)
        expansion_channels = alloc_df[alloc_df["budget_change"] > 500].sort_values(
            by="budget_change", ascending=False
        )
        for _, row in expansion_channels.iterrows():
            ch = row["channel"]
            diff = row["budget_change"]
            diff_pct = row["change_pct"]
            exp_rev = row["expected_revenue"]
            exp_roas = row["expected_roas"]
            m_roas = row["marginal_roas"]

            recommendations.append(
                Recommendation(
                    priority="HIGH",
                    category="Budget Reallocation",
                    target_name=ch,
                    target_type="channel",
                    evidence=(
                        f"Current spend: {format_currency(row['current_budget'])}, "
                        f"Recommended spend: {format_currency(row['recommended_budget'])} (+{diff_pct:.1f}%), "
                        f"Expected ROAS: {format_multiplier(exp_roas)}, Marginal ROAS: {format_multiplier(m_roas)}"
                    ),
                    business_explanation=(
                        f"Channel '{ch}' demonstrates strong marginal capital efficiency with room under the saturation curve. "
                        f"Increasing budget generates an estimated {format_currency(exp_rev)} in attributed revenue."
                    ),
                    recommended_action=(
                        f"Gradually scale daily budget by {format_currency(diff)} (+{diff_pct:.1f}%) across high-performing campaigns, "
                        "monitoring weekly marginal ROAS to prevent premature saturation."
                    ),
                    estimated_budget_impact=f"+{format_currency(diff)} reallocation into high-efficiency channel.",
                    confidence_limitation="Prediction based on fitted diminishing returns response curve and historical conversion rates.",
                )
            )

        # Channels recommended for budget reduction (< -$500 decrease)
        trim_channels = alloc_df[alloc_df["budget_change"] < -500].sort_values(by="budget_change")
        for _, row in trim_channels.iterrows():
            ch = row["channel"]
            diff = abs(row["budget_change"])
            diff_pct = abs(row["change_pct"])
            exp_roas = row["expected_roas"]

            recommendations.append(
                Recommendation(
                    priority="HIGH",
                    category="Budget Reallocation",
                    target_name=ch,
                    target_type="channel",
                    evidence=(
                        f"Current spend: {format_currency(row['current_budget'])}, "
                        f"Recommended: {format_currency(row['recommended_budget'])} (-{diff_pct:.1f}%), "
                        f"Historical ROAS: {format_multiplier(exp_roas)} vs Target {format_multiplier(thresholds.min_roas)}"
                    ),
                    business_explanation=(
                        f"Channel '{ch}' exhibits sub-target capital productivity or high marginal costs. "
                        f"Trimming underperforming spend prevents cash burn and frees budget for higher-yielding channels."
                    ),
                    recommended_action=(
                        f"Reduce channel allocation by {format_currency(diff)} (-{diff_pct:.1f}%), "
                        "pruning unproductive target keywords and ad sets before deploying remaining funds."
                    ),
                    estimated_budget_impact=f"Saves {format_currency(diff)} in inefficient ad expenditure.",
                    confidence_limitation="Assumes trimmed spend came from lowest-performing ad placements within the channel.",
                )
            )

        # Experiment reserve recommendation
        if opt_result.experiment_reserve > 0:
            recommendations.append(
                Recommendation(
                    priority="LOW",
                    category="Experimentation",
                    target_name="Exploratory Reserve Pool",
                    target_type="channel",
                    evidence=f"Experiment reserve: {format_currency(opt_result.experiment_reserve)} ({opt_result.reserve_pct:.1f}% of total budget)",
                    business_explanation=(
                        "Preserving an isolated testing budget ensures continuous discovery of new keywords, audiences, "
                        "and creative formats without risking core return targets."
                    ),
                    recommended_action=(
                        f"Deploy {format_currency(opt_result.experiment_reserve)} strictly toward structured A/B tests "
                        "and emerging channels with capped daily budgets."
                    ),
                    estimated_budget_impact=f"Guards {format_currency(opt_result.experiment_reserve)} from non-experimental deployment.",
                    confidence_limitation="High variance expected; test campaigns should run for at least 14 days before evaluation.",
                )
            )

    # 2. Campaign Level Optimization Recommendations
    if not campaign_df.empty:
        # A. High Clicks but Low Conversion Rate (Listing / PDP / Targeting Inefficiency)
        high_clicks_low_cvr = campaign_df[
            (campaign_df["clicks"] >= 120) &
            (campaign_df["cvr"] < 2.5) &
            (campaign_df["marketing_spend"] >= thresholds.min_spend)
        ].sort_values(by="marketing_spend", ascending=False)

        for _, row in high_clicks_low_cvr.head(3).iterrows():
            cname = row["campaign_name"]
            cvr = row["cvr"]
            clicks = int(row["clicks"])
            spend = row["marketing_spend"]
            convs = int(row["conversions"])

            recommendations.append(
                Recommendation(
                    priority="HIGH",
                    category="Conversion Rate Optimization",
                    target_name=f"{cname} ({row['campaign_id']})",
                    target_type="campaign",
                    evidence=(
                        f"Spend: {format_currency(spend)}, Clicks: {clicks:,}, "
                        f"Conversions: {convs}, Conversion Rate: {format_percentage(cvr)} (benchmark: >4.0%)"
                    ),
                    business_explanation=(
                        f"Campaign '{cname}' successfully drives shopper traffic but fails to convert on Amazon product detail pages. "
                        "This indicates price resistance, negative review sentiment, mismatched search intent, or stock availability issues."
                    ),
                    recommended_action=(
                        "Audit search query reports for irrelevant search terms, add negative keywords, and review Amazon "
                        "product listing image gallery, bullet points, A+ Content, and pricing competitiveness."
                    ),
                    estimated_budget_impact=f"Potential +30-50% conversion lift on {format_currency(spend)} traffic spend.",
                    confidence_limitation="Conversion tracking assumes standard 14-day attribution window.",
                )
            )

        # B. High CPA Outliers with Sufficient Spend
        high_cpa_campaigns = campaign_df[
            (campaign_df["cpa"] > thresholds.max_cpa * 1.35) &
            (campaign_df["conversions"] >= 8) &
            (campaign_df["marketing_spend"] >= thresholds.min_spend)
        ].sort_values(by="cpa", ascending=False)

        for _, row in high_cpa_campaigns.head(3).iterrows():
            cname = row["campaign_name"]
            cpa = row["cpa"]
            spend = row["marketing_spend"]
            roas = row["roas"]

            recommendations.append(
                Recommendation(
                    priority="HIGH" if roas < thresholds.min_roas else "MEDIUM",
                    category="Bidding Strategy",
                    target_name=f"{cname} ({row['campaign_id']})",
                    target_type="campaign",
                    evidence=(
                        f"Observed CPA: {format_currency(cpa)} vs Target {format_currency(thresholds.max_cpa)}, "
                        f"Spend: {format_currency(spend)}, ROAS: {format_multiplier(roas)}"
                    ),
                    business_explanation=(
                        f"Campaign '{cname}' pays an excessive cost per customer acquisition ({format_currency(cpa)}), "
                        f"substantially eroding gross profit contribution."
                    ),
                    recommended_action=(
                        "Lower maximum CPC bids by 15-25%, switch from broad to phrase/exact match keywords, "
                        "and restrict bids on non-converting placements."
                    ),
                    estimated_budget_impact=f"Reduces CPA toward target {format_currency(thresholds.max_cpa)}, saving estimated {format_currency(spend * 0.20)}.",
                    confidence_limitation="Sufficient sample size with multiple verified customer conversions.",
                )
            )

        # C. Insufficient Evidence Campaigns (Test & Learn Recommendation)
        insufficient_campaigns = campaign_df[
            (campaign_df.get("classification") == "Insufficient Data") |
            (campaign_df["conversions"] < 5) & (campaign_df["marketing_spend"] < thresholds.min_spend)
        ]
        if not insufficient_campaigns.empty:
            count = len(insufficient_campaigns)
            tot_spend = insufficient_campaigns["marketing_spend"].sum()
            sample_names = ", ".join(insufficient_campaigns["campaign_name"].head(2).tolist())

            recommendations.append(
                Recommendation(
                    priority="LOW",
                    category="Experimentation",
                    target_name=f"{count} Emerging Campaigns ({sample_names}...)",
                    target_type="campaign",
                    evidence=f"{count} campaigns have under {thresholds.min_conversions} conversions with total spend {format_currency(tot_spend)}.",
                    business_explanation=(
                        "Prematurely pausing or scaling campaigns without statistically sufficient evidence causes either missed growth "
                        "opportunities or unverified budget waste."
                    ),
                    recommended_action=(
                        "Allow campaigns to run with capped daily budgets ($15-$25/day) until at least 30 clicks and 10 conversions "
                        "are recorded before making structural budget reallocation decisions."
                    ),
                    estimated_budget_impact=f"Controls budget risk on {format_currency(tot_spend)} exploratory spend.",
                    confidence_limitation="Observations do not yet achieve statistical significance.",
                )
            )

        # D. High Performing Campaigns with Low Spend (Scale Winners)
        scale_candidates = campaign_df[
            (campaign_df["roas"] >= thresholds.min_roas * 1.3) &
            (campaign_df["conversions"] >= thresholds.min_conversions) &
            (campaign_df["marketing_spend"] < campaign_df["marketing_spend"].median())
        ].sort_values(by="roas", ascending=False)

        for _, row in scale_candidates.head(2).iterrows():
            cname = row["campaign_name"]
            roas = row["roas"]
            spend = row["marketing_spend"]
            rev = row["attributed_revenue"]

            recommendations.append(
                Recommendation(
                    priority="MEDIUM",
                    category="Campaign Efficiency",
                    target_name=f"{cname} ({row['campaign_id']})",
                    target_type="campaign",
                    evidence=(
                        f"ROAS: {format_multiplier(roas)} (Target: {format_multiplier(thresholds.min_roas)}), "
                        f"Current Spend: {format_currency(spend)}, Revenue: {format_currency(rev)}"
                    ),
                    business_explanation=(
                        f"Campaign '{cname}' achieves exceptional return on ad spend but has restricted budget allocation. "
                        "Increasing impression share can unlock additional profitable revenue."
                    ),
                    recommended_action=(
                        "Increase campaign daily budget cap by 25-35% and expand top-of-search placement bid modifiers."
                    ),
                    estimated_budget_impact=f"Estimated incremental revenue of +{format_currency(spend * roas * 0.25)}.",
                    confidence_limitation="Monitored for saturation; return rate may slightly decline as impression share expands.",
                )
            )

    # Sort recommendations by Priority: HIGH -> MEDIUM -> LOW
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))

    return recommendations
