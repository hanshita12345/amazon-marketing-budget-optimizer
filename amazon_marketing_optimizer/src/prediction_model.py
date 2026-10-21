"""Machine learning prediction pipeline for campaign revenue and conversion forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_features,
    split_chronological,
)


@dataclass
class ModelEvaluation:
    """Evaluation metrics comparing model to baseline on out-of-time test set."""
    model_name: str
    target_name: str
    mae: float
    rmse: float
    r2: float
    baseline_mae: float
    baseline_rmse: float
    baseline_r2: float
    mae_improvement_pct: float
    test_samples: int
    train_samples: int


@dataclass
class PredictionArtifacts:
    """Encapsulates trained models, test evaluations, and actual vs predicted frames."""
    revenue_pipeline: Any
    conversion_pipeline: Any
    revenue_eval: ModelEvaluation
    conversion_eval: ModelEvaluation
    test_results_df: pd.DataFrame
    feature_importance_df: Optional[pd.DataFrame] = None
    is_fallback: bool = False
    limitation_notes: List[str] = field(default_factory=list)


def build_pipeline(model_type: str = "random_forest", random_state: int = 42) -> Pipeline:
    """Construct an sklearn Pipeline with ColumnTransformer and Regressor."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    if model_type == "hist_gradient_boosting":
        regressor = HistGradientBoostingRegressor(
            random_state=random_state,
            max_iter=150,
            learning_rate=0.08,
            min_samples_leaf=10,
        )
    else:  # random_forest
        regressor = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Compute MAE, RMSE, and R2 score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # For R2, guard against near-zero variance
    if np.var(y_true) < 1e-6:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def train_and_evaluate_models(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    test_ratio: float = 0.20,
    random_state: int = 42,
) -> PredictionArtifacts:
    """Train revenue and conversion models chronologically and evaluate on held-out test data."""
    notes: List[str] = []

    # Check minimum data requirement
    if len(df) < 50:
        notes.append("Dataset has fewer than 50 rows. ML models require sufficient historical records; using baseline heuristic estimates.")
        return _create_fallback_artifacts(df, notes)

    # 1. Feature Engineering
    feat_df = create_features(df)

    # 2. Chronological Split (prevents future leakage)
    train_df, test_df = split_chronological(feat_df, test_ratio=test_ratio)

    if len(train_df) < 30 or len(test_df) < 10:
        notes.append("Insufficient distinct dates to form stable train/test chronological split; using baseline estimates.")
        return _create_fallback_artifacts(df, notes)

    X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    X_test = test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

    # Revenue target
    y_rev_train = train_df["attributed_revenue"].values
    y_rev_test = test_df["attributed_revenue"].values

    # Conversions target
    y_conv_train = train_df["conversions"].values
    y_conv_test = test_df["conversions"].values

    # 3. Train Revenue Model
    rev_pipeline = build_pipeline(model_type=model_type, random_state=random_state)
    rev_pipeline.fit(X_train, y_rev_train)
    rev_pred_test = np.maximum(0.0, rev_pipeline.predict(X_test))

    # 4. Train Conversion Model
    conv_pipeline = build_pipeline(model_type=model_type, random_state=random_state)
    conv_pipeline.fit(X_train, y_conv_train)
    conv_pred_test = np.maximum(0.0, conv_pipeline.predict(X_test))

    # 5. Baseline Heuristic Predictions (Channel historical average ROAS and CPA on training set)
    channel_roas_map = (
        train_df.groupby("channel")["attributed_revenue"].sum()
        / np.maximum(1.0, train_df.groupby("channel")["marketing_spend"].sum())
    ).to_dict()
    channel_cpa_map = (
        train_df.groupby("channel")["marketing_spend"].sum()
        / np.maximum(1.0, train_df.groupby("channel")["conversions"].sum())
    ).to_dict()

    base_rev_pred = np.array([
        row["marketing_spend"] * channel_roas_map.get(row["channel"], 2.5)
        for _, row in test_df.iterrows()
    ])
    base_conv_pred = np.array([
        row["marketing_spend"] / channel_cpa_map.get(row["channel"], 25.0)
        for _, row in test_df.iterrows()
    ])

    # 6. Evaluation Metrics
    rev_mae, rev_rmse, rev_r2 = compute_metrics(y_rev_test, rev_pred_test)
    base_rev_mae, base_rev_rmse, base_rev_r2 = compute_metrics(y_rev_test, base_rev_pred)
    rev_impr = ((base_rev_mae - rev_mae) / base_rev_mae * 100) if base_rev_mae > 0 else 0.0

    conv_mae, conv_rmse, conv_r2 = compute_metrics(y_conv_test, conv_pred_test)
    base_conv_mae, base_conv_rmse, base_conv_r2 = compute_metrics(y_conv_test, base_conv_pred)
    conv_impr = ((base_conv_mae - conv_mae) / base_conv_mae * 100) if base_conv_mae > 0 else 0.0

    rev_eval = ModelEvaluation(
        model_name=model_type,
        target_name="Attributed Revenue",
        mae=round(rev_mae, 2),
        rmse=round(rev_rmse, 2),
        r2=round(rev_r2, 3),
        baseline_mae=round(base_rev_mae, 2),
        baseline_rmse=round(base_rev_rmse, 2),
        baseline_r2=round(base_rev_r2, 3),
        mae_improvement_pct=round(rev_impr, 1),
        test_samples=len(test_df),
        train_samples=len(train_df),
    )

    conv_eval = ModelEvaluation(
        model_name=model_type,
        target_name="Conversions",
        mae=round(conv_mae, 2),
        rmse=round(conv_rmse, 2),
        r2=round(conv_r2, 3),
        baseline_mae=round(base_conv_mae, 2),
        baseline_rmse=round(base_conv_rmse, 2),
        baseline_r2=round(base_conv_r2, 3),
        mae_improvement_pct=round(conv_impr, 1),
        test_samples=len(test_df),
        train_samples=len(train_df),
    )

    # 7. Build Test Results DataFrame
    results_df = test_df[[
        "date",
        "campaign_id",
        "campaign_name",
        "channel",
        "product_category",
        "marketing_spend",
        "attributed_revenue",
        "conversions",
    ]].copy()

    results_df["predicted_revenue"] = np.round(rev_pred_test, 2)
    results_df["predicted_conversions"] = np.round(conv_pred_test, 1)
    results_df["baseline_predicted_revenue"] = np.round(base_rev_pred, 2)
    results_df["revenue_residual"] = np.round(results_df["attributed_revenue"] - results_df["predicted_revenue"], 2)

    # 8. Feature Importance (if RandomForest)
    feat_imp_df = None
    if model_type == "random_forest":
        try:
            regressor = rev_pipeline.named_steps["regressor"]
            preproc = rev_pipeline.named_steps["preprocessor"]
            cat_names = preproc.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
            feature_names = cat_names + NUMERIC_FEATURES
            importances = regressor.feature_importances_
            feat_imp_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances,
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        except Exception:
            feat_imp_df = None

    return PredictionArtifacts(
        revenue_pipeline=rev_pipeline,
        conversion_pipeline=conv_pipeline,
        revenue_eval=rev_eval,
        conversion_eval=conv_eval,
        test_results_df=results_df,
        feature_importance_df=feat_imp_df,
        is_fallback=False,
        limitation_notes=notes,
    )


def _create_fallback_artifacts(df: pd.DataFrame, notes: List[str]) -> PredictionArtifacts:
    """Generate baseline heuristic evaluation when data is insufficient for machine learning."""
    sample_eval = ModelEvaluation(
        model_name="Historical Baseline Heuristic",
        target_name="Attributed Revenue",
        mae=0.0,
        rmse=0.0,
        r2=0.0,
        baseline_mae=0.0,
        baseline_rmse=0.0,
        baseline_r2=0.0,
        mae_improvement_pct=0.0,
        test_samples=0,
        train_samples=len(df),
    )
    conv_eval = ModelEvaluation(
        model_name="Historical Baseline Heuristic",
        target_name="Conversions",
        mae=0.0,
        rmse=0.0,
        r2=0.0,
        baseline_mae=0.0,
        baseline_rmse=0.0,
        baseline_r2=0.0,
        mae_improvement_pct=0.0,
        test_samples=0,
        train_samples=len(df),
    )
    return PredictionArtifacts(
        revenue_pipeline=None,
        conversion_pipeline=None,
        revenue_eval=sample_eval,
        conversion_eval=conv_eval,
        test_results_df=pd.DataFrame(),
        feature_importance_df=None,
        is_fallback=True,
        limitation_notes=notes,
    )


def predict_candidate_spend(
    artifacts: PredictionArtifacts,
    channel: str,
    product_category: str,
    spend_levels: List[float],
    avg_reference_row: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Predict expected revenue and conversions for varying candidate spend levels on a channel."""
    if artifacts.is_fallback or artifacts.revenue_pipeline is None:
        # Fallback conservative baseline curve
        records = []
        for s in spend_levels:
            # Conservative diminishing returns
            est_rev = s * 2.8 * (1.0 / (1.0 + 0.0003 * s))
            est_conv = s / 28.0
            records.append({
                "marketing_spend": s,
                "predicted_revenue": round(est_rev, 2),
                "predicted_conversions": round(est_conv, 1),
            })
        return pd.DataFrame(records)

    records = []
    for s in spend_levels:
        mock_row = {
            "channel": channel,
            "product_category": product_category,
            "marketing_spend": float(s),
            "day_of_week": 2,  # mid-week representative
            "month": 9,
            "is_weekend": 0,
            "campaign_age_days": 45,
            "rolling_7d_spend": float(s),
            "rolling_7d_revenue": float(s * 3.0),
            "historical_cpc": 1.20,
            "historical_ctr": 0.02,
            "historical_cvr": 0.05,
        }
        if avg_reference_row is not None:
            for k in mock_row:
                if k in avg_reference_row and k not in ["marketing_spend", "channel", "product_category"]:
                    mock_row[k] = avg_reference_row[k]

        input_df = pd.DataFrame([mock_row])
        pred_rev = float(artifacts.revenue_pipeline.predict(input_df)[0])
        pred_conv = float(artifacts.conversion_pipeline.predict(input_df)[0])

        records.append({
            "marketing_spend": s,
            "predicted_revenue": max(0.0, round(pred_rev, 2)),
            "predicted_conversions": max(0.0, round(pred_conv, 1)),
        })

    return pd.DataFrame(records)
