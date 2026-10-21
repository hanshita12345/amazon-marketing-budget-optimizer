"""Amazon AI Marketing Budget Optimization Agent — Streamlit Interactive Web Application.

A comprehensive executive-grade decision support system featuring multi-channel performance
analytics, machine learning forecasting, constrained budget optimization, explainable recommendations,
and automated PDF executive reporting.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Application modules
from src.budget_optimizer import (
    ChannelConstraint,
    OptimizationResult,
    build_channel_curves_from_data,
    optimize_budget,
)
from src.campaign_analysis import (
    PerformanceThresholds,
    analyze_campaign_performance,
    get_performance_distribution,
)
from src.data_loader import get_default_data_path, load_sample_data, load_uploaded_file
from src.data_validation import validate_marketing_data
from src.metrics import (
    calculate_campaign_kpis,
    calculate_channel_kpis,
    calculate_executive_kpis,
    format_currency,
    format_multiplier,
    format_percentage,
)
from src.pdf_report import generate_pdf_report
from src.prediction_model import (
    PredictionArtifacts,
    predict_candidate_spend,
    train_and_evaluate_models,
)
from src.recommendation_engine import Recommendation, generate_recommendations

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon AI Marketing Budget Optimizer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS FOR AMAZON-INSPIRED EXECUTIVE AESTHETICS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global styles */
    .main {
        background-color: #f8f9fa;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .top-header-bar {
        background: linear-gradient(135deg, #131921 0%, #232f3e 100%);
        color: #ffffff;
        padding: 24px 28px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        display: flex;
        flex-direction: column;
    }
    .top-header-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .top-header-badge {
        background-color: #ff9900;
        color: #111111;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
        margin-left: 10px;
    }
    .top-header-subtitle {
        font-size: 14px;
        color: #d5dbdb;
        margin-top: 6px;
    }
    
    /* Executive Metric Card */
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .metric-highlight {
        color: #ff9900;
    }
    .metric-success {
        color: #10b981;
    }
    .metric-danger {
        color: #ef4444;
    }
    
    /* Recommendation Card */
    .rec-card-high {
        background: #ffffff;
        border-left: 5px solid #ef4444;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .rec-card-med {
        background: #ffffff;
        border-left: 5px solid #ff9900;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .rec-card-low {
        background: #ffffff;
        border-left: 5px solid #3b82f6;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .rec-badge-high {
        background: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .rec-badge-med {
        background: #fef3c7;
        color: #92400e;
        font-weight: 700;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .rec-badge-low {
        background: #e0f2fe;
        color: #075985;
        font-weight: 700;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
    }
    
    /* Disclaimer footer */
    .demo-disclaimer {
        font-size: 12px;
        color: #64748b;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# CACHED DATA & MODEL PIPELINE FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_sample_data() -> pd.DataFrame:
    """Load default sample dataset with caching."""
    return load_sample_data(save_if_missing=True)


@st.cache_data
def get_cached_validation(raw_df: pd.DataFrame):
    """Validate dataset and cache result."""
    return validate_marketing_data(raw_df)


@st.cache_resource
def get_cached_ml_models(df: pd.DataFrame, model_type: str) -> PredictionArtifacts:
    """Train machine learning pipelines with resource caching."""
    return train_and_evaluate_models(df, model_type=model_type)


@st.cache_data
def get_cached_curves(df: pd.DataFrame):
    """Fit channel response curves with caching."""
    return build_channel_curves_from_data(df)


# -----------------------------------------------------------------------------
# APPLICATION HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="top-header-bar">
        <div class="top-header-title">
            <span>🛒 Amazon AI Marketing Budget Optimization Agent</span>
            <span class="top-header-badge">AI Agent Decision System</span>
        </div>
        <div class="top-header-subtitle">
            Autonomous multi-channel analytics, Scikit-learn predictive modeling, SciPy constrained budget optimization & executive reporting.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & DATA SOURCE CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/320px-Amazon_logo.svg.png",
        width=130,
    )
    st.subheader("Data & Model Control")

    data_source = st.radio(
        "Data Source",
        ["Default Synthetic Dataset", "Upload CSV File", "Upload Excel (.xlsx)"],
        index=0,
        help="Select default realistic Amazon campaign simulation data or upload your custom dataset.",
    )

    raw_df: Optional[pd.DataFrame] = None
    upload_status_msg = ""

    if data_source == "Default Synthetic Dataset":
        raw_df = get_cached_sample_data()
        st.success(f"Loaded default sample data ({len(raw_df):,} rows).")
    elif data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload Campaign CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = load_uploaded_file(uploaded_file, uploaded_file.name)
                st.success(f"Uploaded '{uploaded_file.name}' ({len(raw_df):,} rows).")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        else:
            st.info("Awaiting CSV upload. Falling back to default data.")
            raw_df = get_cached_sample_data()
    else:  # Upload Excel
        uploaded_file = st.file_uploader("Upload Campaign Excel", type=["xlsx", "xls"])
        if uploaded_file is not None:
            try:
                raw_df = load_uploaded_file(uploaded_file, uploaded_file.name)
                st.success(f"Uploaded '{uploaded_file.name}' ({len(raw_df):,} rows).")
            except Exception as e:
                st.error(f"Error loading Excel: {e}")
        else:
            st.info("Awaiting Excel upload. Falling back to default data.")
            raw_df = get_cached_sample_data()

    # Validate dataset
    val_result = get_cached_validation(raw_df)
    if not val_result.is_valid:
        st.error("Data Validation Failed:")
        for err in val_result.errors:
            st.write(f"- {err}")
        st.stop()

    working_df = val_result.cleaned_df.copy()

    if val_result.warnings:
        with st.expander("Data Quality Warnings", expanded=False):
            for w in val_result.warnings:
                st.write(f"- {w}")

    # Date range selector
    st.markdown("---")
    st.subheader("Reporting Date Range")
    min_date = working_df["date"].min().date()
    max_date = working_df["date"].max().date()

    selected_dates = st.date_input(
        "Date Window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_d, end_d = selected_dates
        working_df = working_df[
            (working_df["date"].dt.date >= start_d) & (working_df["date"].dt.date <= end_d)
        ]
    elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
        start_d = selected_dates[0]
        working_df = working_df[working_df["date"].dt.date >= start_d]

    # Channel filter
    all_channels = sorted(working_df["channel"].unique().tolist())
    selected_channels = st.multiselect(
        "Filter Channels",
        options=all_channels,
        default=all_channels,
    )
    if selected_channels:
        working_df = working_df[working_df["channel"].isin(selected_channels)]

    st.markdown("---")
    # Configurable Classification Thresholds
    with st.expander("Classification Thresholds", expanded=False):
        min_roas_input = st.number_input("Target Minimum ROAS", min_value=1.0, max_value=10.0, value=2.5, step=0.1)
        max_cpa_input = st.number_input("Target Maximum CPA ($)", min_value=5.0, max_value=200.0, value=35.0, step=1.0)
        min_conv_input = st.number_input("Min Conversions for Evidence", min_value=1, max_value=50, value=10, step=1)
        min_spend_input = st.number_input("Min Spend for Evidence ($)", min_value=10.0, max_value=1000.0, value=150.0, step=25.0)

        thresholds = PerformanceThresholds(
            min_roas=min_roas_input,
            max_cpa=max_cpa_input,
            min_conversions=int(min_conv_input),
            min_spend=min_spend_input,
        )


# -----------------------------------------------------------------------------
# CORE ANALYTICAL COMPUTATIONS
# -----------------------------------------------------------------------------
if working_df.empty:
    st.warning("No campaign data available for the selected filters.")
    st.stop()

# Compute KPIs
exec_kpis = calculate_executive_kpis(working_df)
channel_kpis = calculate_channel_kpis(working_df)
analyzed_campaigns = analyze_campaign_performance(working_df, thresholds=thresholds)
perf_distribution = get_performance_distribution(analyzed_campaigns)


# -----------------------------------------------------------------------------
# APPLICATION TABS
# -----------------------------------------------------------------------------
tab_overview, tab_campaigns, tab_predictions, tab_optimizer, tab_recommendations, tab_reports = st.tabs([
    "📊 Executive Overview",
    "🎯 Campaign Analysis",
    "🧠 AI Predictions",
    "⚖️ Budget Optimizer",
    "💡 AI Recommendations",
    "📄 Management Report",
])


# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================================
with tab_overview:
    st.subheader("Executive Financial & Marketing Overview")

    # Metric Cards (Row 1)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Marketing Spend</div>
                <div class="metric-value">{format_currency(exec_kpis['total_spend'])}</div>
                <div class="metric-subtext">{len(working_df):,} daily campaign records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Attributed Revenue</div>
                <div class="metric-value metric-highlight">{format_currency(exec_kpis['total_revenue'])}</div>
                <div class="metric-subtext">Avg Order: {format_currency(exec_kpis['overall_arpc'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        roas_val = exec_kpis['overall_roas']
        roas_class = "metric-success" if roas_val >= thresholds.min_roas else "metric-danger"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Blended ROAS</div>
                <div class="metric-value {roas_class}">{format_multiplier(roas_val)}</div>
                <div class="metric-subtext">Target: {format_multiplier(thresholds.min_roas)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Marketing Contribution</div>
                <div class="metric-value">{format_currency(exec_kpis['marketing_contribution'])}</div>
                <div class="metric-subtext">Gross Profit: {format_currency(exec_kpis['estimated_gross_profit'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Metric Cards (Row 2)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Conversions</div>
                <div class="metric-value">{exec_kpis['total_conversions']:,}</div>
                <div class="metric-subtext">CVR: {exec_kpis['overall_cvr']:.2f}% | CTR: {exec_kpis['overall_ctr']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Average CPA</div>
                <div class="metric-value">{format_currency(exec_kpis['overall_cpa'])}</div>
                <div class="metric-subtext">Average CPC: {format_currency(exec_kpis['overall_cpc'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col7:
        new_cust_str = f"{exec_kpis['total_new_customers']:,}" if exec_kpis['total_new_customers'] else "N/A"
        cac_str = format_currency(exec_kpis['overall_cac']) if exec_kpis['overall_cac'] else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">New Customers Acquired</div>
                <div class="metric-value">{new_cust_str}</div>
                <div class="metric-subtext">CAC: {cac_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col8:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Marketing ROI</div>
                <div class="metric-value">{format_percentage(exec_kpis['overall_roi'])}</div>
                <div class="metric-subtext">Avg Gross Margin: {exec_kpis['weighted_gross_margin']*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts: Spend vs Revenue by Channel & Trend Over Time
    c_chart1, c_chart2 = st.columns([3, 2])
    with c_chart1:
        st.markdown("##### Spend & Attributed Revenue by Channel")
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=channel_kpis["channel"],
            y=channel_kpis["marketing_spend"],
            name="Marketing Spend",
            marker_color="#232f3e",
        ))
        bar_fig.add_trace(go.Bar(
            x=channel_kpis["channel"],
            y=channel_kpis["attributed_revenue"],
            name="Attributed Revenue",
            marker_color="#ff9900",
        ))
        bar_fig.update_layout(
            barmode="group",
            margin=dict(l=20, r=20, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickangle=-25),
            yaxis_title="USD ($)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with c_chart2:
        st.markdown("##### Revenue Share by Channel")
        pie_fig = px.pie(
            channel_kpis,
            names="channel",
            values="attributed_revenue",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        pie_fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=30),
            legend=dict(orientation="v", y=0.5),
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    # Time series trend
    st.markdown("##### Daily Marketing Spend vs Attributed Revenue Trend")
    daily_trend = working_df.groupby("date").agg({
        "marketing_spend": "sum",
        "attributed_revenue": "sum",
    }).reset_index()
    line_fig = px.line(
        daily_trend,
        x="date",
        y=["marketing_spend", "attributed_revenue"],
        labels={"value": "Amount ($)", "variable": "Metric", "date": "Date"},
        color_discrete_map={"marketing_spend": "#232f3e", "attributed_revenue": "#ff9900"},
    )
    line_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="USD ($)",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    )
    st.plotly_chart(line_fig, use_container_width=True)


# =============================================================================
# TAB 2: CAMPAIGN PERFORMANCE ANALYSIS
# =============================================================================
with tab_campaigns:
    st.subheader("Campaign Performance Classification & Efficiency Matrix")

    # Distribution summary counts
    dcol1, dcol2, dcol3, dcol4, dcol5 = st.columns(5)
    dcol1.metric("🟢 High Performing", perf_distribution.get("High Performing", {}).get("count", 0))
    dcol2.metric("🔵 Normal", perf_distribution.get("Performing Normally", {}).get("count", 0))
    dcol3.metric("🟡 Underperforming", perf_distribution.get("Underperforming", {}).get("count", 0))
    dcol4.metric("🔴 Inefficient", perf_distribution.get("Potentially Inefficient", {}).get("count", 0))
    dcol5.metric("⚪ Insufficient Data", perf_distribution.get("Insufficient Data", {}).get("count", 0))

    st.markdown("---")

    # Campaign Efficiency Matrix (Scatter Plot)
    st.markdown("##### Campaign Efficiency Matrix (Spend vs ROAS)")
    scatter_fig = px.scatter(
        analyzed_campaigns,
        x="marketing_spend",
        y="roas",
        size="conversions",
        color="classification",
        hover_name="campaign_name",
        hover_data={"channel": True, "cpa": True, "marketing_contribution": True, "roas": ":.2f"},
        color_discrete_map={
            "High Performing": "#28a745",
            "Performing Normally": "#007bff",
            "Underperforming": "#ffc107",
            "Potentially Inefficient": "#dc3545",
            "Insufficient Data": "#6c757d",
        },
        labels={"marketing_spend": "Marketing Spend ($)", "roas": "Observed ROAS (x)"},
    )
    scatter_fig.add_hline(y=thresholds.min_roas, line_dash="dash", line_color="#ff9900", annotation_text=f"Target ROAS ({thresholds.min_roas}x)")
    scatter_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Detailed Table Filter
    st.markdown("##### Detailed Campaign Performance Table")
    selected_class_filter = st.multiselect(
        "Filter by Classification",
        options=["High Performing", "Performing Normally", "Underperforming", "Potentially Inefficient", "Insufficient Data"],
        default=["High Performing", "Performing Normally", "Underperforming", "Potentially Inefficient", "Insufficient Data"],
    )
    filtered_camps = analyzed_campaigns[analyzed_campaigns["classification"].isin(selected_class_filter)]

    # Format table for clean display
    display_cols = [
        "campaign_id",
        "campaign_name",
        "channel",
        "classification",
        "marketing_spend",
        "attributed_revenue",
        "roas",
        "cpa",
        "cvr",
        "marketing_contribution",
        "classification_reason",
    ]
    st.dataframe(
        filtered_camps[display_cols].style.format({
            "marketing_spend": "${:,.2f}",
            "attributed_revenue": "${:,.2f}",
            "roas": "{:.2f}x",
            "cpa": "${:,.2f}",
            "cvr": "{:.2f}%",
            "marketing_contribution": "${:,.2f}",
        }),
        use_container_width=True,
        height=380,
    )


# =============================================================================
# TAB 3: AI PREDICTIONS
# =============================================================================
with tab_predictions:
    st.subheader("Machine Learning Prediction Pipeline (Scikit-learn)")
    st.markdown(
        "Dual-model regression pipeline estimating future attributed revenue and conversions while strictly "
        "preventing future data leakage via chronological train/test splitting."
    )

    p_col1, p_col2 = st.columns([1, 3])
    with p_col1:
        chosen_model = st.selectbox(
            "Model Architecture",
            options=["random_forest", "hist_gradient_boosting"],
            format_func=lambda x: "Random Forest Regressor" if x == "random_forest" else "Histogram Gradient Boosting",
        )
        test_ratio_slider = st.slider("Test Set Split Ratio", 0.10, 0.35, 0.20, 0.05)

    with st.spinner("Training predictive models on chronological historical splits..."):
        ml_artifacts = train_and_evaluate_models(working_df, model_type=chosen_model, test_ratio=test_ratio_slider)

    if ml_artifacts.is_fallback:
        st.warning("⚠️ Baseline Heuristic Active")
        for note in ml_artifacts.limitation_notes:
            st.info(note)
    else:
        # Evaluation Cards
        rev_ev = ml_artifacts.revenue_eval
        conv_ev = ml_artifacts.conversion_eval

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Revenue MAE", f"${rev_ev.mae:,.2f}", f"{rev_ev.mae_improvement_pct:+.1f}% vs baseline", delta_color="inverse")
        m_col2.metric("Revenue RMSE", f"${rev_ev.rmse:,.2f}")
        m_col3.metric("Revenue R² Score", f"{rev_ev.r2:.3f}")
        m_col4.metric("Test Samples", f"{rev_ev.test_samples:,} dates")

        st.markdown("---")

        # Actual vs Predicted Plots
        p_chart1, p_chart2 = st.columns(2)
        test_df = ml_artifacts.test_results_df

        with p_chart1:
            st.markdown("##### Out-of-Time Test Set: Actual vs Predicted Revenue")
            fig_pred_rev = px.scatter(
                test_df,
                x="attributed_revenue",
                y="predicted_revenue",
                color="channel",
                hover_name="campaign_name",
                labels={"attributed_revenue": "Actual Revenue ($)", "predicted_revenue": "Predicted Revenue ($)"},
            )
            # Add 45-degree reference line
            max_val = max(test_df["attributed_revenue"].max(), test_df["predicted_revenue"].max())
            fig_pred_rev.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="#64748b", dash="dash"),
            ))
            fig_pred_rev.update_layout(plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_pred_rev, use_container_width=True)

        with p_chart2:
            st.markdown("##### Actual vs Predicted Revenue Over Time (Test Period)")
            test_ts = test_df.groupby("date").agg({
                "attributed_revenue": "sum",
                "predicted_revenue": "sum",
                "baseline_predicted_revenue": "sum",
            }).reset_index()

            fig_ts = px.line(
                test_ts,
                x="date",
                y=["attributed_revenue", "predicted_revenue", "baseline_predicted_revenue"],
                labels={"value": "Revenue ($)", "variable": "Series", "date": "Date"},
                color_discrete_map={
                    "attributed_revenue": "#0f172a",
                    "predicted_revenue": "#ff9900",
                    "baseline_predicted_revenue": "#94a3b8",
                },
            )
            fig_ts.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # Feature Importance if available
        if ml_artifacts.feature_importance_df is not None:
            st.markdown("##### Top Predictive Features (Random Forest Feature Importance)")
            top_feats = ml_artifacts.feature_importance_df.head(10)
            fig_feat = px.bar(
                top_feats,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig_feat.update_layout(
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_feat, use_container_width=True)


# =============================================================================
# TAB 4: BUDGET OPTIMIZER
# =============================================================================
with tab_optimizer:
    st.subheader("Constrained Marketing Budget Optimizer (SciPy SLSQP)")
    st.markdown(
        "Mathematically allocates available capital to maximize estimated incremental marketing contribution, "
        "rigorously modeling diminishing marginal returns and channel saturation."
    )

    curves = get_cached_curves(working_df)
    current_total_spend = float(working_df["marketing_spend"].sum())

    # User inputs for Budget Optimization
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        total_budget_input = st.number_input(
            "Total Marketing Budget ($)",
            min_value=1000.0,
            max_value=2000000.0,
            value=float(round(current_total_spend, -2)),
            step=5000.0,
            help="Total capital available to deploy across channels.",
        )
    with b_col2:
        reserve_pct_input = st.slider(
            "Exploratory Reserve Pool (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            help="Reserve dedicated to structured experimentation and emerging channels.",
        )
    with b_col3:
        target_min_roas_input = st.number_input(
            "Target Minimum Portfolio ROAS",
            min_value=1.0,
            max_value=8.0,
            value=float(thresholds.min_roas),
            step=0.1,
            help="Soft constraint enforcing portfolio blended return target.",
        )

    # Interactive Channel Bounds
    st.markdown("##### Channel Allocation Constraints (Min / Max Bounds)")
    with st.expander("Adjust Channel Bounds & Constraints", expanded=False):
        channel_constraints_list: List[ChannelConstraint] = []
        bound_cols = st.columns(len(curves))
        for i, (ch, curve) in enumerate(curves.items()):
            c_spend = curve.historical_spend
            with bound_cols[i % len(bound_cols)]:
                st.markdown(f"**{ch}**")
                min_b = st.number_input(
                    f"Min Spend ($)",
                    min_value=0.0,
                    max_value=float(c_spend * 2.0),
                    value=float(round(c_spend * 0.40, -1)),
                    key=f"min_{ch}",
                )
                max_b = st.number_input(
                    f"Max Spend ($)",
                    min_value=float(min_b),
                    max_value=float(c_spend * 5.0),
                    value=float(round(c_spend * 2.50, -1)),
                    key=f"max_{ch}",
                )
                channel_constraints_list.append(
                    ChannelConstraint(
                        channel=ch,
                        current_spend=c_spend,
                        min_spend=min_b,
                        max_spend=max_b,
                        gross_margin_rate=curve.gross_margin_rate,
                        historical_roas=curve.historical_revenue / max(1.0, c_spend),
                    )
                )

    if not channel_constraints_list:
        # Default fallback constraints
        for ch, curve in curves.items():
            channel_constraints_list.append(
                ChannelConstraint(
                    channel=ch,
                    current_spend=curve.historical_spend,
                    min_spend=curve.historical_spend * 0.35,
                    max_spend=curve.historical_spend * 2.5,
                    gross_margin_rate=curve.gross_margin_rate,
                    historical_roas=curve.historical_revenue / max(1.0, curve.historical_spend),
                )
            )

    run_opt = st.button("🚀 Run Budget Optimization", type="primary", use_container_width=True)

    # Run optimizer (cached or fresh on button click)
    opt_result = optimize_budget(
        total_budget=total_budget_input,
        channel_constraints=channel_constraints_list,
        curves=curves,
        experiment_reserve_pct=reserve_pct_input / 100.0,
        min_target_roas=target_min_roas_input,
    )
    st.session_state["latest_opt_result"] = opt_result

    if opt_result.infeasibility_notes:
        with st.expander("Optimizer Notes & Constraint Adjustments", expanded=True):
            for note in opt_result.infeasibility_notes:
                st.info(f"ℹ️ {note}")

    # Optimization Result Summary Cards
    st.markdown("---")
    st.markdown("##### Optimization Financial Impact")
    o_col1, o_col2, o_col3, o_col4 = st.columns(4)
    o_col1.metric("Allocatable Budget", format_currency(opt_result.allocatable_budget), f"Reserve: {format_currency(opt_result.experiment_reserve)}")
    o_col2.metric("Expected Revenue", format_currency(opt_result.recommended_expected_revenue), f"+{format_currency(opt_result.incremental_revenue)} lift")
    o_col3.metric("Expected Contribution", format_currency(opt_result.recommended_expected_contribution), f"+{format_currency(opt_result.incremental_contribution)} lift")
    o_col4.metric("Expected Portfolio ROAS", format_multiplier(opt_result.overall_expected_roas))

    # Allocation comparison table
    st.markdown("##### Reallocation Comparison by Channel")
    alloc_table = opt_result.channel_allocations.copy()
    st.dataframe(
        alloc_table[[
            "channel",
            "current_budget",
            "recommended_budget",
            "budget_change",
            "change_pct",
            "expected_revenue",
            "expected_contribution",
            "expected_roas",
            "marginal_roas",
        ]].style.format({
            "current_budget": "${:,.2f}",
            "recommended_budget": "${:,.2f}",
            "budget_change": "${:+,.2f}",
            "change_pct": "{:+.1f}%",
            "expected_revenue": "${:,.2f}",
            "expected_contribution": "${:,.2f}",
            "expected_roas": "{:.2f}x",
            "marginal_roas": "{:.2f}x",
        }),
        use_container_width=True,
    )

    # Reallocation Waterfall / Bar Chart
    st.markdown("##### Budget Shift by Channel ($ Change)")
    shift_fig = px.bar(
        alloc_table,
        x="channel",
        y="budget_change",
        color="budget_change",
        color_continuous_scale=["#ef4444", "#e2e8f0", "#10b981"],
        labels={"budget_change": "Recommended Budget Shift ($)", "channel": "Channel"},
    )
    shift_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(shift_fig, use_container_width=True)

    # Sensitivity Curve
    st.markdown("##### Portfolio Sensitivity Analysis (Varying Total Marketing Budget)")
    sens_fig = px.line(
        opt_result.sensitivity_curve,
        x="total_budget",
        y=["expected_revenue", "expected_contribution"],
        markers=True,
        labels={"value": "Projected Amount ($)", "total_budget": "Total Budget ($)", "variable": "Outcome"},
        color_discrete_map={"expected_revenue": "#ff9900", "expected_contribution": "#232f3e"},
    )
    sens_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(sens_fig, use_container_width=True)


# =============================================================================
# TAB 5: AI RECOMMENDATIONS
# =============================================================================
with tab_recommendations:
    st.subheader("AI-Driven Prioritized Strategic Recommendations")
    st.markdown(
        "Deterministic, explainable business recommendations generated directly from empirical KPIs, "
        "campaign classifications, and optimizer allocation shifts."
    )

    active_opt = st.session_state.get("latest_opt_result", opt_result)
    recommendations_list = generate_recommendations(
        campaign_df=analyzed_campaigns,
        channel_df=channel_kpis,
        opt_result=active_opt,
        thresholds=thresholds,
    )

    # Priority Filter
    priority_filter = st.radio("Filter Priority", ["All", "HIGH", "MEDIUM", "LOW"], horizontal=True)
    if priority_filter != "All":
        display_recs = [r for r in recommendations_list if r.priority == priority_filter]
    else:
        display_recs = recommendations_list

    st.markdown(f"Showing **{len(display_recs)}** prioritized recommendations:")

    for rec in display_recs:
        card_class = "rec-card-high" if rec.priority == "HIGH" else ("rec-card-med" if rec.priority == "MEDIUM" else "rec-card-low")
        badge_class = "rec-badge-high" if rec.priority == "HIGH" else ("rec-badge-med" if rec.priority == "MEDIUM" else "rec-badge-low")

        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span class="{badge_class}">{rec.priority} PRIORITY</span>
                        <span style="font-size: 11px; font-weight: 600; color: #64748b; margin-left: 8px;">{rec.category.upper()}</span>
                    </div>
                    <span style="font-size: 13px; font-weight: 700; color: #0f172a;">{rec.target_name}</span>
                </div>
                <div style="font-size: 13px; font-weight: 600; color: #1e293b; margin-bottom: 6px;">
                    🎯 Recommended Action: {rec.recommended_action}
                </div>
                <div style="font-size: 12px; color: #475569; margin-bottom: 6px;">
                    <b>Business Context:</b> {rec.business_explanation}
                </div>
                <div style="font-size: 12px; color: #0369a1; background: #f0f9ff; padding: 6px 10px; border-radius: 4px; margin-bottom: 6px;">
                    <b>📊 Empirical Evidence:</b> {rec.evidence}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; border-top: 1px solid #f1f5f9; padding-top: 6px;">
                    <span><b>Estimated Impact:</b> {rec.estimated_budget_impact}</span>
                    <span><b>Modeling Note:</b> {rec.confidence_limitation}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 6: MANAGEMENT REPORT & EXPORTS
# =============================================================================
with tab_reports:
    st.subheader("Executive Management Report & Data Export")
    st.markdown(
        "Generate a publication-grade PDF management report formatted for executive leadership, "
        "or download raw analyzed data tables for external reporting."
    )

    active_opt = st.session_state.get("latest_opt_result", opt_result)
    under_camps = analyzed_campaigns[analyzed_campaigns["classification"].isin(["Underperforming", "Potentially Inefficient"])]

    r_col1, r_col2 = st.columns([2, 3])
    with r_col1:
        st.markdown("##### Generate PDF Management Report")
        st.write(
            "Compiles executive summary, multi-channel KPI tables, underperforming campaign audits, "
            "predictive model governance, budget reallocation schedule, and prioritized recommendations."
        )

        if st.button("📑 Generate Official PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating ReportLab executive PDF document..."):
                try:
                    pdf_path, pdf_bytes = generate_pdf_report(
                        exec_kpis=exec_kpis,
                        channel_kpis=channel_kpis,
                        underperforming_campaigns=under_camps,
                        opt_result=active_opt,
                        recommendations=recommendations_list,
                        prediction_eval={
                            "model_name": chosen_model.replace("_", " ").title(),
                            "mae": ml_artifacts.revenue_eval.mae if not ml_artifacts.is_fallback else "N/A",
                            "rmse": ml_artifacts.revenue_eval.rmse if not ml_artifacts.is_fallback else "N/A",
                            "r2": ml_artifacts.revenue_eval.r2 if not ml_artifacts.is_fallback else "N/A",
                            "improvement_pct": ml_artifacts.revenue_eval.mae_improvement_pct if not ml_artifacts.is_fallback else 0.0,
                            "test_samples": ml_artifacts.revenue_eval.test_samples if not ml_artifacts.is_fallback else 0,
                        },
                    )
                    st.session_state["generated_pdf_bytes"] = pdf_bytes
                    st.session_state["generated_pdf_path"] = pdf_path
                    st.success("PDF Report generated successfully!")
                except Exception as e:
                    st.error(f"PDF Generation Error: {str(e)}")

        if "generated_pdf_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Executive PDF Report",
                data=st.session_state["generated_pdf_bytes"],
                file_name=f"Amazon_Marketing_Budget_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with r_col2:
        st.markdown("##### Download Analytical CSV Data")
        st.write("Export clean campaign records, channel aggregations, and optimization schedules.")

        csv_c1, csv_c2 = st.columns(2)
        with csv_c1:
            # Download analyzed campaign data
            camp_csv = analyzed_campaigns.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Analyzed Campaigns (CSV)",
                data=camp_csv,
                file_name="analyzed_campaign_kpis.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Download channel metrics
            chan_csv = channel_kpis.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Channel Performance (CSV)",
                data=chan_csv,
                file_name="channel_performance_kpis.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with csv_c2:
            # Download recommended budget allocations
            if active_opt is not None and not active_opt.channel_allocations.empty:
                alloc_csv = active_opt.channel_allocations.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Budget Reallocations (CSV)",
                    data=alloc_csv,
                    file_name="recommended_budget_allocation.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Download sensitivity table
            if active_opt is not None and not active_opt.sensitivity_curve.empty:
                sens_csv = active_opt.sensitivity_curve.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Sensitivity Analysis (CSV)",
                    data=sens_csv,
                    file_name="budget_sensitivity_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# -----------------------------------------------------------------------------
# APPLICATION FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="demo-disclaimer">
        <b>Academic & Professional Demonstration Notice:</b> This is an independent marketing decision system
        demonstration project developed using synthetic e-commerce data and user-provided inputs. It does not
        represent an official Amazon application and does not have access to Amazon's proprietary internal marketing
        databases. Any live ad budget modifications require authorized human governance and approval.
    </div>
    """,
    unsafe_allow_html=True,
)
