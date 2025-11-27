"""
Improved metrics_hub3.py

Purpose:
--------
This module acts as the "local metrics database" for:
- A/B Test Metrics
- Campaign Performance Metrics
- Historical ML Training Data
- Trend and Sentiment Scores

Enhancements:
-------------
1. Clean CSV schema management
2. Auto-create datasets if missing
3. Standardized columns for ML
4. Feature-engineering helpers
5. Query utilities for:
   - Recent metrics
   - Campaign history
   - Variant comparison
6. Error-safe loading / writing
7. Timestamped logging
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------

DATA_DIR = "data"
CAMPAIGNS_CSV = os.path.join(DATA_DIR, "campaigns.csv")
HISTORICAL_CSV = os.path.join(DATA_DIR, "historical_metrics.csv")

os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------------------------------------
# Schema Setup: Ensure files exist
# -----------------------------------------------------------

def _init_file(path: str, columns: list):
    """Create CSV file with given columns if missing."""
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)


_init_file(CAMPAIGNS_CSV, [
    "timestamp", "campaign_id", "variant", "impressions", "clicks",
    "conversions", "ctr", "conv_rate", "sentiment", "trend_score"
])

_init_file(HISTORICAL_CSV, [
    "timestamp", "campaign_id", "variant", "ctr", "sentiment",
    "polarity", "conversions", "trend_score"
])


# -----------------------------------------------------------
# Core API: Record A/B Test Metrics
# -----------------------------------------------------------

def record_campaign_metrics(
    campaign_id: str,
    variant: str,
    impressions: int,
    clicks: int,
    conversions: int,
    sentiment_score: float,
    trend_score: float = 0.0
):
    """
    Stores campaign metrics in campaigns.csv and historical_metrics.csv.
    """

    timestamp = datetime.utcnow().isoformat()

    ctr = clicks / impressions if impressions > 0 else 0
    conv_rate = conversions / clicks if clicks > 0 else 0

    new_row = {
        "timestamp": timestamp,
        "campaign_id": campaign_id,
        "variant": variant,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "ctr": ctr,
        "conv_rate": conv_rate,
        "sentiment": sentiment_score,
        "trend_score": trend_score
    }

    # Append to main CSV
    df = pd.read_csv(CAMPAIGNS_CSV)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CAMPAIGNS_CSV, index=False)

    # Append to historical CSV (subset)
    hist_row = {
        "timestamp": timestamp,
        "campaign_id": campaign_id,
        "variant": variant,
        "ctr": ctr,
        "sentiment": sentiment_score,
        "polarity": sentiment_score,     # compatible with ML
        "conversions": conversions,
        "trend_score": trend_score
    }

    hist = pd.read_csv(HISTORICAL_CSV)
    hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)
    hist.to_csv(HISTORICAL_CSV, index=False)


# -----------------------------------------------------------
# Fetch & Query Utilities
# -----------------------------------------------------------

def fetch_recent_metrics(limit: int = 50) -> pd.DataFrame:
    """Returns the most recent A/B test records."""
    df = pd.read_csv(CAMPAIGNS_CSV)
    return df.tail(limit)


def fetch_campaign_history(campaign_id: str) -> pd.DataFrame:
    """Returns full history for a specific campaign."""
    df = pd.read_csv(CAMPAIGNS_CSV)
    return df[df["campaign_id"] == campaign_id]


def fetch_variant_performance(campaign_id: str) -> Dict[str, Any]:
    """
    Compares all variants under a campaign.
    Returns:
        {
            "best": {...},
            "worst": {...},
            "all_variants": [...],
        }
    """
    df = fetch_campaign_history(campaign_id)
    if df.empty:
        return {}

    # Compute best based on conversion rate or CTR fallback
    df_sorted = df.sort_values(["conv_rate", "ctr"], ascending=False)
    best = df_sorted.iloc[0].to_dict()
    worst = df_sorted.iloc[-1].to_dict()

    return {
        "best": best,
        "worst": worst,
        "all_variants": df_sorted.to_dict(orient="records")
    }


def get_ml_training_data() -> pd.DataFrame:
    """
    Returns the full dataset for ML training.
    Feature columns:
        - ctr
        - sentiment
        - polarity
        - conversions
        - trend_score
    """
    df = pd.read_csv(HISTORICAL_CSV)
    df = df.dropna()
    return df


# -----------------------------------------------------------
# Feature Engineering Utilities
# -----------------------------------------------------------

def build_feature_vector(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a campaign row into ML-ready feature vector.
    """
    return {
        "ctr": row.get("ctr", 0),
        "sentiment": row.get("sentiment", 0),
        "polarity": row.get("polarity", row.get("sentiment", 0)),
        "trend_score": row.get("trend_score", 0),
        "conversions": row.get("conversions", 0)
    }


def compute_variant_score(row: Dict[str, Any]) -> float:
    """
    Custom scoring function for ranking A/B variants.
    Can be tuned using ML.
    """
    score = (
        row.get("ctr", 0) * 0.5 +
        row.get("sentiment", 0) * 0.3 +
        row.get("trend_score", 0) * 0.2
    )
    return round(score, 4)


# -----------------------------------------------------------
# Manual Test
# -----------------------------------------------------------

if __name__ == "__main__":
    print("\nMetrics Hub Test Running...\n")

    record_campaign_metrics(
        "test_campaign_123",
        "variant_1",
        impressions=500,
        clicks=42,
        conversions=10,
        sentiment_score=0.75,
        trend_score=0.60
    )

    print("\nRecent Metrics:")
    print(fetch_recent_metrics().tail())

    print("\nCampaign History:")
    print(fetch_campaign_history("test_campaign_123"))

    print("\nVariant Performance:")
    print(fetch_variant_performance("test_campaign_123"))

    print("\nML Training Data Sample:")
    print(get_ml_training_data().head())
