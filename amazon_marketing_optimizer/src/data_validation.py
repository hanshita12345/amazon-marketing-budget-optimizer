"""Data validation and quality assessment module for marketing campaign datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Canonical column names required for full analysis
REQUIRED_COLUMNS = [
    "date",
    "campaign_id",
    "campaign_name",
    "channel",
    "marketing_spend",
    "impressions",
    "clicks",
    "conversions",
    "attributed_revenue",
]

OPTIONAL_COLUMNS = [
    "product_category",
    "gross_margin_rate",
    "new_customers",
    "returning_customers",
]

# Alias dictionary to support user-uploaded schemas with different naming conventions
COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["date", "day", "report_date", "timestamp", "dt"],
    "campaign_id": ["campaign_id", "cid", "id", "campaignid", "campaign_code"],
    "campaign_name": ["campaign_name", "campaign", "cname", "name", "campaign_title"],
    "channel": ["channel", "ad_channel", "source", "marketing_channel", "medium", "network"],
    "product_category": ["product_category", "category", "product_cat", "department"],
    "marketing_spend": ["marketing_spend", "spend", "cost", "ad_spend", "marketing_cost", "expenditure", "budget_spent"],
    "impressions": ["impressions", "imprs", "views", "impressions_count"],
    "clicks": ["clicks", "ad_clicks", "user_clicks"],
    "conversions": ["conversions", "orders", "purchases", "convs", "sales_count"],
    "attributed_revenue": ["attributed_revenue", "revenue", "sales", "attributed_sales", "conv_value", "total_sales"],
    "gross_margin_rate": ["gross_margin_rate", "margin", "gross_margin", "margin_rate", "profit_margin"],
    "new_customers": ["new_customers", "new_cust", "new_buyers", "first_time_customers"],
    "returning_customers": ["returning_customers", "returning_cust", "repeat_buyers", "repeat_customers"],
}


@dataclass
class ValidationResult:
    """Encapsulates validation outcome, errors, warnings, and sanitized DataFrame."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_df: Optional[pd.DataFrame] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
        }


def map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Map user DataFrame columns to canonical schema using fuzzy/alias mapping.

    Returns:
        renamed_df: DataFrame with canonical column names
        mapping_applied: Dict showing original -> canonical name
        missing_required: List of required canonical columns not matched
    """
    col_mapping: Dict[str, str] = {}
    normalized_cols = {col.strip().lower().replace(" ", "_"): col for col in df.columns}

    # Match canonical keys
    for canonical, aliases in COLUMN_ALIASES.items():
        matched_original = None
        for alias in aliases:
            norm_alias = alias.strip().lower().replace(" ", "_")
            if norm_alias in normalized_cols:
                matched_original = normalized_cols[norm_alias]
                break
        if matched_original:
            col_mapping[matched_original] = canonical

    renamed_df = df.rename(columns=col_mapping)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in renamed_df.columns]
    mapping_applied = {orig: can for orig, can in col_mapping.items() if orig != can}

    return renamed_df, mapping_applied, missing_required


def validate_marketing_data(df: pd.DataFrame) -> ValidationResult:
    """Validate and sanitize a marketing dataset.

    Performs exhaustive checks on required columns, data types, value boundaries,
    logical relationships (e.g., clicks <= impressions), duplicates, and null values.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return ValidationResult(
            is_valid=False,
            errors=["Uploaded dataset is empty or invalid."],
            warnings=[],
            cleaned_df=None,
            summary={"initial_rows": 0, "valid_rows": 0},
        )

    initial_row_count = len(df)

    # 1. Column Mapping & Schema Check
    mapped_df, mapping_applied, missing_required = map_columns(df.copy())
    if mapping_applied:
        for orig, can in mapping_applied.items():
            warnings.append(f"Column '{orig}' was mapped to standard schema '{can}'.")

    if missing_required:
        errors.append(
            f"Missing required columns: {', '.join(missing_required)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            cleaned_df=None,
            summary={"initial_rows": initial_row_count, "missing_columns": missing_required},
        )

    # Work on a clean copy
    clean = mapped_df.copy()

    # 2. Date parsing and validation
    try:
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
        invalid_dates = clean["date"].isna().sum()
        if invalid_dates > 0:
            errors.append(f"Found {invalid_dates} rows with unparseable or missing date values.")
    except Exception as e:
        errors.append(f"Date conversion error: {str(e)}")

    # 3. String columns sanitization
    for str_col in ["campaign_id", "campaign_name", "channel"]:
        clean[str_col] = clean[str_col].astype(str).str.strip()
        empty_str_count = (clean[str_col] == "").sum() + (clean[str_col].str.lower() == "nan").sum()
        if empty_str_count > 0:
            warnings.append(f"Found {empty_str_count} rows with blank/missing '{str_col}'.")

    # Optional product_category default if missing
    if "product_category" not in clean.columns:
        clean["product_category"] = "General"
        warnings.append("Optional column 'product_category' missing; defaulted to 'General'.")
    else:
        clean["product_category"] = clean["product_category"].fillna("General").astype(str).str.strip()

    # 4. Numeric conversion and boundary checks
    numeric_cols = [
        "marketing_spend",
        "impressions",
        "clicks",
        "conversions",
        "attributed_revenue",
    ]

    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        null_count = clean[col].isna().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} missing or non-numeric values.")

        neg_count = (clean[col] < 0).sum()
        if neg_count > 0:
            errors.append(f"Column '{col}' contains {neg_count} invalid negative values.")

    # Integer checks for counts
    for count_col in ["impressions", "clicks", "conversions"]:
        if count_col in clean.columns and clean[count_col].notna().all():
            clean[count_col] = clean[count_col].round().astype(int)

    # 5. Logical consistency checks
    # Clicks cannot exceed impressions
    invalid_clicks = (clean["clicks"] > clean["impressions"]).sum()
    if invalid_clicks > 0:
        errors.append(f"Logical error in {invalid_clicks} rows: clicks exceed impressions.")

    # Conversions cannot exceed clicks
    invalid_convs = (clean["conversions"] > clean["clicks"]).sum()
    if invalid_convs > 0:
        errors.append(f"Logical error in {invalid_convs} rows: conversions exceed clicks.")

    # 6. Gross margin rate handling
    if "gross_margin_rate" not in clean.columns:
        clean["gross_margin_rate"] = 0.35
        warnings.append(
            "Optional column 'gross_margin_rate' was not provided. "
            "A default baseline of 0.35 (35%) has been applied for gross profit estimation."
        )
    else:
        clean["gross_margin_rate"] = pd.to_numeric(clean["gross_margin_rate"], errors="coerce")
        # Check if user entered percentage as 0-100 instead of 0-1
        high_margins = (clean["gross_margin_rate"] > 1.0).sum()
        if high_margins > 0:
            clean.loc[clean["gross_margin_rate"] > 1.0, "gross_margin_rate"] /= 100.0
            warnings.append(
                f"Detected {high_margins} rows with gross margin > 1.0; "
                "normalized from percentage (0-100%) to decimal (0.0-1.0)."
            )
        invalid_margin = ((clean["gross_margin_rate"] < 0) | (clean["gross_margin_rate"] > 1.0)).sum()
        if invalid_margin > 0:
            errors.append(f"Found {invalid_margin} rows with gross margin rate outside valid [0.0, 1.0] range.")
        if clean["gross_margin_rate"].isna().any():
            clean["gross_margin_rate"] = clean["gross_margin_rate"].fillna(0.35)
            warnings.append("Replaced missing gross margin values with baseline 0.35.")

    # 7. Customer acquisition counts
    if "new_customers" not in clean.columns:
        clean["new_customers"] = np.nan
        warnings.append("Optional column 'new_customers' missing; Customer Acquisition Cost (CAC) will be calculated where available.")
    else:
        clean["new_customers"] = pd.to_numeric(clean["new_customers"], errors="coerce")
        invalid_new = (clean["new_customers"] > clean["conversions"]).sum()
        if invalid_new > 0:
            warnings.append(f"{invalid_new} rows had new_customers exceeding conversions; clamped to conversion count.")
            clean["new_customers"] = np.minimum(clean["new_customers"], clean["conversions"])

    if "returning_customers" not in clean.columns:
        if clean["new_customers"].notna().all():
            clean["returning_customers"] = clean["conversions"] - clean["new_customers"]
        else:
            clean["returning_customers"] = np.nan
    else:
        clean["returning_customers"] = pd.to_numeric(clean["returning_customers"], errors="coerce")

    # 8. Duplicate detection
    dup_mask = clean.duplicated(subset=["date", "campaign_id"], keep=False)
    dup_count = dup_mask.sum()
    if dup_count > 0:
        warnings.append(
            f"Found {dup_count} duplicate records for the same (date, campaign_id). "
            "Aggregated duplicate campaign entries by summing metrics."
        )
        agg_rules = {
            "campaign_name": "first",
            "channel": "first",
            "product_category": "first",
            "marketing_spend": "sum",
            "impressions": "sum",
            "clicks": "sum",
            "conversions": "sum",
            "attributed_revenue": "sum",
            "gross_margin_rate": "mean",
        }
        if clean["new_customers"].notna().any():
            agg_rules["new_customers"] = "sum"
        if clean["returning_customers"].notna().any():
            agg_rules["returning_customers"] = "sum"

        clean = clean.groupby(["date", "campaign_id"], as_index=False).agg(agg_rules)

    # Sort clean data
    clean = clean.sort_values(by=["date", "channel", "campaign_id"]).reset_index(drop=True)

    is_valid = len(errors) == 0

    summary = {
        "initial_rows": initial_row_count,
        "valid_rows": len(clean) if is_valid else 0,
        "date_min": clean["date"].min().strftime("%Y-%m-%d") if (is_valid and clean["date"].notna().any()) else None,
        "date_max": clean["date"].max().strftime("%Y-%m-%d") if (is_valid and clean["date"].notna().any()) else None,
        "channels": clean["channel"].unique().tolist() if is_valid else [],
        "campaigns_count": clean["campaign_id"].nunique() if is_valid else 0,
        "total_spend": float(clean["marketing_spend"].sum()) if is_valid else 0.0,
        "total_revenue": float(clean["attributed_revenue"].sum()) if is_valid else 0.0,
    }

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        cleaned_df=clean if is_valid else None,
        summary=summary,
    )
