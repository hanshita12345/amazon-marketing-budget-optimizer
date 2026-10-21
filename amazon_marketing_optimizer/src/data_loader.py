"""Data loader and synthetic dataset generation module for Amazon Marketing Optimizer.

NOTE: This module generates realistic synthetic demonstration data for an Amazon-inspired
e-commerce business case study. It is an independent demonstration and does not use or claim
access to Amazon's private internal marketing data.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union
import numpy as np
import pandas as pd


CHANNEL_CONFIGS = {
    "Amazon Sponsored Products": {
        "cpc_range": (0.95, 1.45),
        "cvr_range": (0.055, 0.095),
        "spend_range": (70.0, 160.0),
        "aov_range": (38.0, 68.0),
        "ctr_range": (0.009, 0.024),
    },
    "Amazon Sponsored Brands": {
        "cpc_range": (1.20, 1.80),
        "cvr_range": (0.040, 0.075),
        "spend_range": (60.0, 140.0),
        "aov_range": (48.0, 88.0),
        "ctr_range": (0.007, 0.018),
    },
    "Amazon Sponsored Display": {
        "cpc_range": (0.85, 1.35),
        "cvr_range": (0.020, 0.045),
        "spend_range": (50.0, 120.0),
        "aov_range": (42.0, 75.0),
        "ctr_range": (0.005, 0.014),
    },
    "Google Ads": {
        "cpc_range": (1.35, 2.20),
        "cvr_range": (0.035, 0.065),
        "spend_range": (60.0, 130.0),
        "aov_range": (50.0, 90.0),
        "ctr_range": (0.015, 0.035),
    },
    "Meta Ads": {
        "cpc_range": (1.10, 1.90),
        "cvr_range": (0.018, 0.042),
        "spend_range": (50.0, 110.0),
        "aov_range": (44.0, 80.0),
        "ctr_range": (0.009, 0.022),
    },
    "Email Marketing": {
        "cpc_range": (0.35, 0.70),
        "cvr_range": (0.045, 0.085),
        "spend_range": (15.0, 40.0),
        "aov_range": (45.0, 78.0),
        "ctr_range": (0.025, 0.055),
    },
    "Affiliate Marketing": {
        "cpc_range": (0.75, 1.25),
        "cvr_range": (0.038, 0.070),
        "spend_range": (30.0, 75.0),
        "aov_range": (46.0, 82.0),
        "ctr_range": (0.012, 0.028),
    },
}

PRODUCT_CATEGORIES = {
    "Electronics": {"margin_range": (0.22, 0.30), "aov_mult": 1.30},
    "Home & Kitchen": {"margin_range": (0.32, 0.42), "aov_mult": 1.08},
    "Beauty & Personal Care": {"margin_range": (0.42, 0.54), "aov_mult": 0.88},
    "Apparel": {"margin_range": (0.38, 0.50), "aov_mult": 0.95},
    "Sports & Outdoors": {"margin_range": (0.30, 0.42), "aov_mult": 1.15},
    "Books & Media": {"margin_range": (0.24, 0.35), "aov_mult": 0.65},
}


def generate_synthetic_dataset(num_days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate realistic Amazon marketing campaign dataset."""
    rng = np.random.default_rng(seed)

    campaign_definitions = [
        # Amazon Sponsored Products
        ("CMP-SP-001", "Echo_SmartHome_AutoKW", "Amazon Sponsored Products", "Electronics", "normal"),
        ("CMP-SP-002", "FireTV_Streaming_Exact", "Amazon Sponsored Products", "Electronics", "high"),
        ("CMP-SP-003", "Cookware_NonStick_Broad", "Amazon Sponsored Products", "Home & Kitchen", "normal"),
        ("CMP-SP-004", "AirFryer_Pro_Competitor", "Amazon Sponsored Products", "Home & Kitchen", "inefficient"),
        ("CMP-SP-005", "Skincare_Serum_HighIntent", "Amazon Sponsored Products", "Beauty & Personal Care", "high"),
        ("CMP-SP-006", "YogaMat_EcoGrip_Phrase", "Amazon Sponsored Products", "Sports & Outdoors", "normal"),
        ("CMP-SP-007", "RunningShoes_Lightweight", "Amazon Sponsored Products", "Apparel", "normal"),
        ("CMP-SP-008", "KindleEbooks_Bestsellers", "Amazon Sponsored Products", "Books & Media", "insufficient"),

        # Amazon Sponsored Brands
        ("CMP-SB-001", "BrandStore_LivingRoom_Video", "Amazon Sponsored Brands", "Home & Kitchen", "high"),
        ("CMP-SB-002", "Audio_NoiseCancelling_Headline", "Amazon Sponsored Brands", "Electronics", "normal"),
        ("CMP-SB-003", "Activewear_Collection_Spotlight", "Amazon Sponsored Brands", "Apparel", "inefficient"),
        ("CMP-SB-004", "OrganicBeauty_Store_Header", "Amazon Sponsored Brands", "Beauty & Personal Care", "normal"),
        ("CMP-SB-005", "OutdoorGear_Adventure_Brand", "Amazon Sponsored Brands", "Sports & Outdoors", "insufficient"),

        # Amazon Sponsored Display
        ("CMP-SD-001", "Retargeting_Electronics_Viewers", "Amazon Sponsored Display", "Electronics", "normal"),
        ("CMP-SD-002", "CrossSell_KitchenAppliances", "Amazon Sponsored Display", "Home & Kitchen", "inefficient"),
        ("CMP-SD-003", "Audience_Fitness_Health", "Amazon Sponsored Display", "Sports & Outdoors", "normal"),
        ("CMP-SD-004", "CompetitorASIN_Beauty_Defense", "Amazon Sponsored Display", "Beauty & Personal Care", "inefficient"),

        # Google Ads
        ("CMP-GOOG-001", "Google_Search_Smart_Displays", "Google Ads", "Electronics", "high"),
        ("CMP-GOOG-002", "Google_Shopping_Kitchen_Gadgets", "Google Ads", "Home & Kitchen", "normal"),
        ("CMP-GOOG-003", "Google_Search_Winter_Jackets", "Google Ads", "Apparel", "inefficient"),
        ("CMP-GOOG-004", "Google_NonBrand_Skincare_Routine", "Google Ads", "Beauty & Personal Care", "normal"),

        # Meta Ads
        ("CMP-META-001", "Instagram_Reels_KitchenHacks", "Meta Ads", "Home & Kitchen", "normal"),
        ("CMP-META-002", "Facebook_Feed_SmartHome_Promo", "Meta Ads", "Electronics", "inefficient"),
        ("CMP-META-003", "Instagram_Story_Athleisure_Launch", "Meta Ads", "Apparel", "normal"),
        ("CMP-META-004", "Meta_Lookalike_Beauty_VIP", "Meta Ads", "Beauty & Personal Care", "insufficient"),

        # Email Marketing
        ("CMP-EML-001", "Weekly_Deals_Prime_Subscribers", "Email Marketing", "Electronics", "high"),
        ("CMP-EML-002", "Loyalty_VIP_HomeDecor_Reengage", "Email Marketing", "Home & Kitchen", "high"),
        ("CMP-EML-003", "Beauty_Refill_Automated_Drip", "Email Marketing", "Beauty & Personal Care", "high"),

        # Affiliate Marketing
        ("CMP-AFF-001", "TechReviewer_Affiliate_Network", "Affiliate Marketing", "Electronics", "high"),
        ("CMP-AFF-002", "LifestyleBloggers_Curated_Picks", "Affiliate Marketing", "Home & Kitchen", "normal"),
        ("CMP-AFF-003", "FitnessCoupons_Affiliate_Deals", "Affiliate Marketing", "Sports & Outdoors", "normal"),
    ]

    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=num_days - 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    records = []

    for dt in date_range:
        day_of_week = dt.dayofweek
        weekend_boost = 1.15 if day_of_week in [5, 6] else 1.0

        for cid, cname, channel, category, archetype in campaign_definitions:
            cfg = CHANNEL_CONFIGS[channel]
            cat_cfg = PRODUCT_CATEGORIES[category]

            if archetype == "insufficient":
                if rng.random() > 0.35:
                    continue
                daily_spend = rng.uniform(4.0, 16.0)
                perf_mult = rng.uniform(0.75, 1.10)
            elif archetype == "high":
                sp_low, sp_high = cfg["spend_range"]
                daily_spend = rng.uniform(sp_low * 1.1, sp_high * 1.3) * weekend_boost
                perf_mult = rng.uniform(1.25, 1.55)
            elif archetype == "inefficient":
                sp_low, sp_high = cfg["spend_range"]
                daily_spend = rng.uniform(sp_low * 1.2, sp_high * 1.4) * weekend_boost
                perf_mult = rng.uniform(0.40, 0.65)
            else:  # normal
                sp_low, sp_high = cfg["spend_range"]
                daily_spend = rng.uniform(sp_low, sp_high) * weekend_boost
                perf_mult = rng.uniform(0.92, 1.15)

            cpc_low, cpc_high = cfg["cpc_range"]
            if archetype == "inefficient":
                cpc_low *= 1.25
                cpc_high *= 1.45
            elif archetype == "high":
                cpc_low *= 0.88
                cpc_high *= 0.95

            cpc = rng.uniform(cpc_low, cpc_high)
            clicks = int(np.maximum(1, np.round(daily_spend / cpc)))

            ctr_low, ctr_high = cfg["ctr_range"]
            ctr = rng.uniform(ctr_low, ctr_high) * (1.15 if archetype == "high" else (0.85 if archetype == "inefficient" else 1.0))
            impressions = int(np.maximum(clicks + 15, np.round(clicks / ctr)))

            cvr_low, cvr_high = cfg["cvr_range"]
            cvr = rng.uniform(cvr_low, cvr_high) * perf_mult
            conversions = int(np.round(clicks * cvr))
            if conversions > clicks:
                conversions = clicks

            if conversions > 0:
                aov_low, aov_high = cfg["aov_range"]
                base_aov = rng.uniform(aov_low, aov_high) * cat_cfg["aov_mult"]
                attributed_revenue = round(conversions * base_aov * rng.uniform(0.93, 1.07), 2)
            else:
                attributed_revenue = 0.0

            m_low, m_high = cat_cfg["margin_range"]
            gross_margin_rate = round(rng.uniform(m_low, m_high), 3)

            if conversions > 0:
                new_ratio = 0.65 if channel in ["Meta Ads", "Google Ads"] else 0.42
                if channel == "Email Marketing":
                    new_ratio = 0.20
                new_cust = int(np.round(conversions * rng.uniform(new_ratio - 0.08, new_ratio + 0.08)))
                new_customers = np.clip(new_cust, 0, conversions)
                returning_customers = conversions - new_customers
            else:
                new_customers = 0
                returning_customers = 0

            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "campaign_id": cid,
                "campaign_name": cname,
                "channel": channel,
                "product_category": category,
                "marketing_spend": round(float(daily_spend), 2),
                "impressions": int(impressions),
                "clicks": int(clicks),
                "conversions": int(conversions),
                "attributed_revenue": float(attributed_revenue),
                "gross_margin_rate": float(gross_margin_rate),
                "new_customers": int(new_customers),
                "returning_customers": int(returning_customers),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["date", "channel", "campaign_id"]).reset_index(drop=True)
    return df


def get_default_data_path() -> Path:
    """Return default path to sample campaign dataset."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "data" / "sample_campaign_data.csv"


def load_sample_data(save_if_missing: bool = True, force_regenerate: bool = False) -> pd.DataFrame:
    """Load default sample campaign data, generating it if not present."""
    data_path = get_default_data_path()
    if data_path.exists() and not force_regenerate:
        df = pd.read_csv(data_path)
        return df

    df = generate_synthetic_dataset(num_days=90, seed=42)
    if save_if_missing:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    return df


def load_uploaded_file(file_obj: Union[BinaryIO, bytes, Path, str], filename: str) -> pd.DataFrame:
    """Load user uploaded CSV or Excel file safely."""
    fname = filename.lower()
    if fname.endswith(".csv"):
        if isinstance(file_obj, (str, Path)):
            return pd.read_csv(file_obj)
        elif isinstance(file_obj, bytes):
            return pd.read_csv(io.BytesIO(file_obj))
        else:
            return pd.read_csv(file_obj)
    elif fname.endswith((".xlsx", ".xls")):
        if isinstance(file_obj, (str, Path)):
            return pd.read_excel(file_obj)
        elif isinstance(file_obj, bytes):
            return pd.read_excel(io.BytesIO(file_obj))
        else:
            return pd.read_excel(file_obj)
    else:
        raise ValueError(f"Unsupported file format '{filename}'. Please upload a .csv or .xlsx file.")
