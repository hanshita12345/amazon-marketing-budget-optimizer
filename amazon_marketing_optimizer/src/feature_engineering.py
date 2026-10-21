"""Feature engineering pipeline for marketing campaign machine learning models."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = ["channel", "product_category"]
NUMERIC_FEATURES = [
    "marketing_spend",
    "day_of_week",
    "month",
    "is_weekend",
    "campaign_age_days",
    "rolling_7d_spend",
    "rolling_7d_revenue",
    "historical_cpc",
    "historical_ctr",
    "historical_cvr",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer robust temporal, historical, and efficiency features for ML prediction.

    Ensures chronological ordering so rolling and historical metrics do not leak future information.
    """
    if df.empty:
        return df.copy()

    feat_df = df.copy()
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    feat_df = feat_df.sort_values(by=["campaign_id", "date"]).reset_index(drop=True)

    # 1. Calendar and Seasonality features
    feat_df["day_of_week"] = feat_df["date"].dt.dayofweek
    feat_df["day_of_month"] = feat_df["date"].dt.day
    feat_df["month"] = feat_df["date"].dt.month
    feat_df["is_weekend"] = feat_df["day_of_week"].isin([5, 6]).astype(int)

    # 2. Campaign age in days (from first observation in dataset)
    first_date_map = feat_df.groupby("campaign_id")["date"].transform("min")
    feat_df["campaign_age_days"] = (feat_df["date"] - first_date_map).dt.days

    # 3. Rolling 7-day metrics (shifted by 1 to strictly avoid target leakage)
    # Group by campaign_id and compute shifted rolling averages
    feat_df["rolling_7d_spend"] = (
        feat_df.groupby("campaign_id")["marketing_spend"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
    )
    feat_df["rolling_7d_revenue"] = (
        feat_df.groupby("campaign_id")["attributed_revenue"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
    )

    # 4. Historical efficiency metrics (cumulative lagged to avoid leakage)
    cum_spend = feat_df.groupby("campaign_id")["marketing_spend"].transform(lambda s: s.shift(1).cumsum())
    cum_clicks = feat_df.groupby("campaign_id")["clicks"].transform(lambda s: s.shift(1).cumsum())
    cum_imprs = feat_df.groupby("campaign_id")["impressions"].transform(lambda s: s.shift(1).cumsum())
    cum_convs = feat_df.groupby("campaign_id")["conversions"].transform(lambda s: s.shift(1).cumsum())

    feat_df["historical_cpc"] = np.where(cum_clicks > 0, cum_spend / cum_clicks, np.nan)
    feat_df["historical_ctr"] = np.where(cum_imprs > 0, cum_clicks / cum_imprs, np.nan)
    feat_df["historical_cvr"] = np.where(cum_clicks > 0, cum_convs / cum_clicks, np.nan)

    # Fill initial NaN from shifted metrics with channel median or sensible defaults
    channel_cpc = feat_df.groupby("channel")["marketing_spend"].sum() / feat_df.groupby("channel")["clicks"].sum()
    channel_ctr = feat_df.groupby("channel")["clicks"].sum() / feat_df.groupby("channel")["impressions"].sum()
    channel_cvr = feat_df.groupby("channel")["conversions"].sum() / feat_df.groupby("channel")["clicks"].sum()

    for ch in feat_df["channel"].unique():
        ch_mask = feat_df["channel"] == ch
        feat_df.loc[ch_mask & feat_df["historical_cpc"].isna(), "historical_cpc"] = channel_cpc.get(ch, 1.20)
        feat_df.loc[ch_mask & feat_df["historical_ctr"].isna(), "historical_ctr"] = channel_ctr.get(ch, 0.02)
        feat_df.loc[ch_mask & feat_df["historical_cvr"].isna(), "historical_cvr"] = channel_cvr.get(ch, 0.05)
        feat_df.loc[ch_mask & feat_df["rolling_7d_spend"].isna(), "rolling_7d_spend"] = feat_df.loc[ch_mask, "marketing_spend"].mean()
        feat_df.loc[ch_mask & feat_df["rolling_7d_revenue"].isna(), "rolling_7d_revenue"] = feat_df.loc[ch_mask, "attributed_revenue"].mean()

    # Any remaining NaNs
    feat_df["historical_cpc"] = feat_df["historical_cpc"].fillna(1.20)
    feat_df["historical_ctr"] = feat_df["historical_ctr"].fillna(0.02)
    feat_df["historical_cvr"] = feat_df["historical_cvr"].fillna(0.05)
    feat_df["rolling_7d_spend"] = feat_df["rolling_7d_spend"].fillna(feat_df["marketing_spend"])
    feat_df["rolling_7d_revenue"] = feat_df["rolling_7d_revenue"].fillna(feat_df["attributed_revenue"])

    # Ensure categorical columns are strings
    for col in CATEGORICAL_FEATURES:
        feat_df[col] = feat_df[col].astype(str)

    # Sort back chronologically
    feat_df = feat_df.sort_values(by=["date", "channel", "campaign_id"]).reset_index(drop=True)
    return feat_df


def split_chronological(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically based on distinct dates to avoid future data leakage.

    The first (1 - test_ratio) dates form the training set; the remaining dates form the test set.
    """
    if "date" not in df.columns or df.empty:
        raise ValueError("DataFrame must contain a 'date' column for chronological splitting.")

    dates = pd.to_datetime(df["date"]).drop_duplicates().sort_values().reset_index(drop=True)
    if len(dates) < 5:
        # Too few distinct dates; fallback to simple 80/20 index split
        split_idx = int(len(df) * (1 - test_ratio))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    split_date_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates.iloc[split_date_idx]

    train_df = df[pd.to_datetime(df["date"]) < split_date].copy()
    test_df = df[pd.to_datetime(df["date"]) >= split_date].copy()

    return train_df, test_df
