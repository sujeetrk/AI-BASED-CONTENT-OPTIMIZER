"""
Test Script for:
- app/metrics_engine/metrics_tracker2.py
- app/metrics_engine/tracker2.py
- app/metrics_engine/metrics_hub3.py

Run using:
    python test_metrics_engine.py
"""

import pandas as pd
from app.metrics_engine.metrics_tracker2 import update_google_sheet, compute_metrics_from_df
from app.metrics_engine.tracker2 import (
    push_raw_feedback,
    push_aggregates,
    push_ab_test_results,
    log_campaign_event
)
from app.metrics_engine.metrics_hub2 import (
    record_campaign_metrics,
    fetch_recent_metrics,
    fetch_campaign_history,
    fetch_variant_performance,
    get_ml_training_data
)

from app.sentiment_engine.sentiment import SentimentResult


# -------------------------------------------------------
# 1. Test: KPI Metrics Calculation (metrics_tracker2)
# -------------------------------------------------------

def test_metrics_tracker():
    print("\n==========================")
    print("🔹 TEST 1: metrics_tracker2")
    print("==========================")

    df = pd.DataFrame({
        "views": [150, 200, 300],
        "likes": [10, 25, 45],
        "clicks": [5, 15, 20],
        "conversions": [1, 3, 5],
        "sentiment_label": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "sentiment_score": [0.8, 0.2, 0.55],
        "polarity": [0.75, -0.4, 0.1],
        "toxicity": [0.02, 0.4, 0.1],
        "emotions": [
            {"joy": 0.8},
            {"anger": 0.7},
            {"sadness": 0.5}
        ]
    })

    metrics = compute_metrics_from_df(df)
    print("\nComputed Metrics:\n", metrics)

    print("\nTesting Google Sheet push (will auto-skip in offline mode)...")
    update_google_sheet("test_metrics_sheet", df)


# -------------------------------------------------------
# 2. Test: Raw Logger + A/B Logger (tracker2)
# -------------------------------------------------------

def test_tracker2():
    print("\n====================")
    print("🔹 TEST 2: tracker2")
    print("====================")

    # Fake sentiment results to test raw feedback logging
    sample_feedback = [
        SentimentResult(
            id="1",
            text="AI tools are amazing!",
            label="POSITIVE",
            score=0.92,
            polarity=0.88,
            toxicity=0.01,
            emotions={"joy": 0.9},
            sarcasm=False,
            aspects={"ai tools": 0.7},
            source="twitter"
        ),
        SentimentResult(
            id="2",
            text="This feature is terrible 😡",
            label="NEGATIVE",
            score=0.10,
            polarity=-0.65,
            toxicity=0.40,
            emotions={"anger": 0.8},
            sarcasm=False,
            aspects={"feature": -0.5},
            source="instagram"
        ),
    ]

    print("\nPushing raw feedback...")
    push_raw_feedback(sample_feedback)

    # Aggregate metrics example
    metrics = {
        "total": 2,
        "avg_score": 0.51,
        "pos_count": 1,
        "neg_count": 1,
        "neu_count": 0,
        "pct_positive": 0.5,
        "pct_negative": 0.5,
        "avg_toxicity": 0.205,
        "dominant_emotion": "joy"
    }

    print("\nPushing aggregated metrics...")
    push_aggregates(metrics)

    # A/B test results example
    print("\nPushing A/B Test results...")
    push_ab_test_results("campaign_demo_01", [
        {"variant": "v1", "impressions": 1000, "clicks": 80, "conversions": 10, "ctr": 0.08, "conv_rate": 0.12},
        {"variant": "v2", "impressions": 1000, "clicks": 60, "conversions": 8, "ctr": 0.06, "conv_rate": 0.13}
    ])

    print("\nLogging campaign event...")
    log_campaign_event("A/B test started", {"campaign": "campaign_demo_01", "variants": 2})


# -------------------------------------------------------
# 3. Test: Metrics Hub (local CSV database)
# -------------------------------------------------------

def test_metrics_hub():
    print("\n===========================")
    print("🔹 TEST 3: metrics_hub3")
    print("===========================")

    # Add new campaign metrics
    print("\nRecording campaign metrics...")
    record_campaign_metrics(
        "cmp_test_123",
        "variant_A",
        impressions=500,
        clicks=50,
        conversions=12,
        sentiment_score=0.78,
        trend_score=0.65
    )

    print("\nRecent metrics:")
    print(fetch_recent_metrics().tail())

    print("\nCampaign history:")
    print(fetch_campaign_history("cmp_test_123"))

    print("\nVariant performance summary:")
    print(fetch_variant_performance("cmp_test_123"))

    print("\nML Training dataset preview:")
    print(get_ml_training_data().head())


# -------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------

if __name__ == "__main__":
    test_metrics_tracker()
    test_tracker2()
    test_metrics_hub()

    print("\n==============================")
    print("🎉 ALL METRICS MODULES TESTED!")
    print("==============================")
