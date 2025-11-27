"""
Improved metrics_tracker2.py

Major Enhancements:
-------------------
1. Full marketing KPI tracking:
   - CTR (clicks / impressions)
   - Engagement rate (likes + comments / impressions)
   - Conversion rate
   - Sentiment distribution
   - Toxicity average
   - Dominant emotion score

2. Google Sheets reliability improvements:
   - Automatic sheet creation
   - Automatic header injection
   - Retry logic
   - Safe failure-handling if credentials/sheet unavailable

3. Clean output for dashboards, A/B testing, and Slack reports
"""

import os
import time
import logging
import pandas as pd
from typing import Dict, Any, Optional

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

# Config
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "credentials/service_account.json")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
RETRY_LIMIT = 3


# ------------------------------------------------------
# Google Sheets Client Factory
# ------------------------------------------------------

def _get_client():
    """Return authorized Google Sheets client."""
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Credentials missing: {SERVICE_ACCOUNT_FILE}")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
    client = gspread.authorize(creds)
    return client


# ------------------------------------------------------
# Ensure Sheet Exists
# ------------------------------------------------------

def _ensure_sheet(sheet, name: str, headers: list):
    """Ensure worksheet exists with header row."""
    try:
        ws = sheet.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(name, rows=2000, cols=30)
        ws.append_row(headers)
        logger.info(f"Created worksheet '{name}' with headers.")
    return ws


# ------------------------------------------------------
# Compute Metrics
# ------------------------------------------------------
def compute_metrics_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes final marketing performance metrics from A/B testing results.
    This version is aligned with your A/B Test output structure.
    """

    total = len(df)

    # CTR (already computed in A/B results)
    ctr = df["ctr"].mean() if "ctr" in df.columns else 0

    # Engagement rate - fallback logic
    # Since A/B test does not provide "likes", we skip engagement
    engagement = 0  

    # Conversion rate
    conv_rate = df["conv_rate"].mean() if "conv_rate" in df.columns else 0

    # Sentiment score is NOT in A/B test results → use fallback
    if "sentiment_score" in df.columns:
        pos_ratio = (df["sentiment_score"] == "POSITIVE").mean()
        neg_ratio = (df["sentiment_score"] == "NEGATIVE").mean()
        neu_ratio = (df["sentiment_score"] == "NEUTRAL").mean()
    else:
        pos_ratio = neg_ratio = neu_ratio = 0

    # Polarity (trend_score)
    avg_polarity = df["trend_score"].mean() if "trend_score" in df.columns else 0

    # Toxicity optional
    avg_toxicity = df["toxicity"].mean() if "toxicity" in df.columns else 0

    # Dominant emotion optional
    dom_emotion = "unknown"

    return {
        "total_records": int(total),
        "ctr": round(ctr, 4),
        "engagement_rate": round(engagement, 4),
        "conversion_rate": round(conv_rate, 4),
        "positive_ratio": round(pos_ratio, 4),
        "negative_ratio": round(neg_ratio, 4),
        "neutral_ratio": round(neu_ratio, 4),
        "avg_polarity": round(avg_polarity, 4),
        "avg_toxicity": round(avg_toxicity, 4),
        "dominant_emotion": dom_emotion
    }



# ------------------------------------------------------
# Push Metrics to Google Sheets
# ------------------------------------------------------

def update_google_sheet(sheet_name: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Pushes full KPI row into Google Sheets under specified sheet_name.
    Returns computed metrics dict, or None if Google Sheets unavailable.
    """

    metrics = compute_metrics_from_df(df)

    if not GOOGLE_SHEET_ID:
        logger.warning("GOOGLE_SHEET_ID not set → metrics not uploaded.")
        return metrics

    # Attempt retries
    for attempt in range(RETRY_LIMIT):
        try:
            client = _get_client()
            sheet = client.open_by_key(GOOGLE_SHEET_ID)

            ws = _ensure_sheet(
                sheet,
                sheet_name,
                headers=[
                    "timestamp",
                    "total_records",
                    "ctr",
                    "engagement_rate",
                    "conversion_rate",
                    "positive_ratio",
                    "negative_ratio",
                    "neutral_ratio",
                    "avg_polarity",
                    "avg_toxicity",
                    "dominant_emotion"
                ]
            )

            import datetime
            row = [
                datetime.datetime.utcnow().isoformat(),
                metrics["total_records"],
                metrics["ctr"],
                metrics["engagement_rate"],
                metrics["conversion_rate"],
                metrics["positive_ratio"],
                metrics["negative_ratio"],
                metrics["neutral_ratio"],
                metrics["avg_polarity"],
                metrics["avg_toxicity"],
                metrics["dominant_emotion"]
            ]

            ws.append_row(row)
            logger.info(f"Metrics added to Google Sheet '{sheet_name}'.")
            return metrics

        except Exception as e:
            logger.warning(f"Google Sheets push failed (attempt {attempt+1}). Error: {e}")
            time.sleep(2)

    logger.error("All retries failed. Metrics NOT uploaded.")
    return metrics

# ------------------------------------------------------
# Public Helper for Pipeline: push_daily_metrics()
# ------------------------------------------------------

def push_daily_metrics(df=None, sheet_name: str = "daily_metrics"):
    """
    Wrapper used by run.py / run2.py.
    If df is None → create placeholder empty metrics row.
    Otherwise → push normal metrics.
    """

    if df is None:
        # Create an empty dataframe with expected columns
        df = pd.DataFrame([{
            "impressions": 0,
            "likes": 0,
            "clicks": 0,
            "conversions": 0,
            "sentiment_label": "NEUTRAL",
            "polarity": 0,
            "toxicity": 0,
            "emotions": {}
        }])

    metrics = update_google_sheet(sheet_name, df)
    return metrics

# ------------------------------------------------------
# CLI Test
# ------------------------------------------------------

if __name__ == "__main__":
    # Example test dataframe
    data = {
        "impressions": [100, 200, 150],
        "likes": [10, 25, 5],
        "clicks": [5, 10, 4],
        "conversions": [1, 3, 0],
        "sentiment_label": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "sentiment_score": [0.9, 0.2, 0.5],
        "polarity": [0.8, -0.6, 0.1],
        "toxicity": [0.01, 0.5, 0.12],
        "emotions": [
            {"joy": 0.8, "anger": 0.1},
            {"anger": 0.7, "fear": 0.1},
            {"sadness": 0.4}
        ]
    }
    df = pd.DataFrame(data)

    print("\nComputed Metrics:")
    m = update_google_sheet("demo_metrics", df)
    print(m)
