"""Unit tests for explainable recommendation engine, ML pipeline, and PDF generation."""

from __future__ import annotations

import pandas as pd
import pytest
from src.budget_optimizer import ChannelConstraint, fit_channel_response_curve, optimize_budget
from src.campaign_analysis import analyze_campaign_performance
from src.data_loader import generate_synthetic_dataset
from src.metrics import calculate_campaign_kpis, calculate_channel_kpis, calculate_executive_kpis
from src.pdf_report import generate_pdf_report
from src.prediction_model import train_and_evaluate_models
from src.recommendation_engine import Recommendation, generate_recommendations


def test_recommendation_generation_and_structure():
    """Verify that recommendations are generated with all required fields and correct priority sorting."""
    df = generate_synthetic_dataset(num_days=45, seed=42)
    camp_kpis = analyze_campaign_performance(df)
    chan_kpis = calculate_channel_kpis(df)

    curves = {}
    constraints = []
    for _, row in chan_kpis.iterrows():
        ch = row["channel"]
        sp = float(row["marketing_spend"])
        rev = float(row["attributed_revenue"])
        margin = float(row.get("gross_margin_rate", 0.35))
        curves[ch] = fit_channel_response_curve(ch, sp, rev, margin)
        constraints.append(
            ChannelConstraint(
                channel=ch,
                current_spend=sp,
                min_spend=sp * 0.4,
                max_spend=sp * 2.2,
                gross_margin_rate=margin,
            )
        )

    opt_res = optimize_budget(
        total_budget=float(df["marketing_spend"].sum()),
        channel_constraints=constraints,
        curves=curves,
    )

    recs = generate_recommendations(camp_kpis, chan_kpis, opt_res)

    assert len(recs) > 0

    # Verify priority sorting: HIGH must precede MEDIUM, which must precede LOW
    priority_ranks = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ranks = [priority_ranks.get(r.priority, 3) for r in recs]
    assert ranks == sorted(ranks), "Recommendations must be sorted by priority."

    # Verify each recommendation has complete metadata
    for r in recs:
        assert r.priority in ["HIGH", "MEDIUM", "LOW"]
        assert len(r.target_name) > 0
        assert len(r.category) > 0
        assert len(r.evidence) > 0
        assert len(r.business_explanation) > 0
        assert len(r.recommended_action) > 0
        assert len(r.estimated_budget_impact) > 0
        assert len(r.confidence_limitation) > 0


def test_ml_pipeline_training_and_evaluation():
    """Verify Scikit-learn predictive model training and evaluation metrics."""
    df = generate_synthetic_dataset(num_days=60, seed=99)
    artifacts = train_and_evaluate_models(df, model_type="random_forest", test_ratio=0.20)

    assert artifacts.is_fallback is False
    assert artifacts.revenue_pipeline is not None
    assert artifacts.conversion_pipeline is not None

    rev_ev = artifacts.revenue_eval
    assert rev_ev.mae > 0.0
    assert rev_ev.rmse > 0.0
    assert rev_ev.test_samples > 0
    assert rev_ev.train_samples > 0
    assert not artifacts.test_results_df.empty
    assert "predicted_revenue" in artifacts.test_results_df.columns


def test_pdf_report_generation():
    """Verify that end-to-end PDF report generation completes without exceptions."""
    df = generate_synthetic_dataset(num_days=30, seed=77)
    exec_kpis = calculate_executive_kpis(df)
    chan_kpis = calculate_channel_kpis(df)
    camp_kpis = analyze_campaign_performance(df)
    under = camp_kpis[camp_kpis["classification"].isin(["Underperforming", "Potentially Inefficient"])]

    recs = [
        Recommendation(
            priority="HIGH",
            category="Budget Reallocation",
            target_name="Amazon Sponsored Products",
            target_type="channel",
            evidence="ROAS 4.2x vs Target 2.5x",
            business_explanation="Strong performance with high marginal returns.",
            recommended_action="Scale daily budget by $2,000.",
            estimated_budget_impact="+$2,000 reallocation",
            confidence_limitation="Fitted curve model.",
        )
    ]

    filepath, pdf_bytes = generate_pdf_report(
        exec_kpis=exec_kpis,
        channel_kpis=chan_kpis,
        underperforming_campaigns=under,
        opt_result=None,
        recommendations=recs,
    )

    assert len(pdf_bytes) > 1000
    assert filepath.endswith(".pdf")
