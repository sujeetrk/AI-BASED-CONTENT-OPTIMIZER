"""
Improved tracker2.py (patched)

Purpose:
--------
Push raw sentiment records + structured feedback + A/B test results
into Google Sheets with safety guards for TextBlob / custom objects.

This version fixes recursion errors when TextBlob objects appear in
results (aspects/emotions) by converting them to plain strings safely.
"""

import os
import gspread
import logging
import datetime
import pandas as pd
import json
from typing import List, Dict, Any
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    logger.addHandler(h)

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file"
]
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "credentials/service_account.json")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
MAX_BATCH = 50


# -----------------------------------------------------------
# Core Helpers
# -----------------------------------------------------------

def _client():
    """Return gspread client, or None if credentials not found."""
    if not GOOGLE_SHEET_ID:
        logger.warning("GOOGLE_SHEET_ID not set → skipping Google Sheets upload.")
        return None
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.warning("Service account credentials missing — offline mode.")
        return None

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return gspread.authorize(creds)


def _get_or_create_ws(sheet, name: str, headers: List[str]):
    """
    Ensures worksheet exists. If not, creates it with headers.
    """
    try:
        ws = sheet.worksheet(name)
        return ws
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(name, rows=2000, cols=30)
        ws.append_row(headers)
        logger.info(f"Created worksheet '{name}' with headers.")
        return ws


# -----------------------------------------------------------
# Serialization Helpers (safe)
# -----------------------------------------------------------

def safe_serialize(obj: Any) -> str:
    """
    Convert common objects to a JSON-friendly string:
    - If obj is a dict/list/str/int/float/bool -> JSON dump (compact)
    - If obj has .string attribute (TextBlob), use .string
    - Otherwise, fallback to str(obj)
    We use json.dumps(..., default=str) to ensure numpy/scalars are handled.
    """
    try:
        # If it's already a JSON-serializable primitive/container
        if isinstance(obj, (dict, list, tuple, str, int, float, bool, type(None))):
            # Convert tuples to lists for JSON friendliness
            if isinstance(obj, tuple):
                obj = list(obj)
            return json.dumps(obj, default=str)
        # TextBlob / spaCy-like objects often expose `.string` or `.text`
        if hasattr(obj, "string"):
            try:
                return json.dumps(str(obj.string), default=str)
            except Exception:
                return json.dumps(str(obj), default=str)
        if hasattr(obj, "text"):
            try:
                return json.dumps(str(obj.text), default=str)
            except Exception:
                return json.dumps(str(obj), default=str)
        # Last resort: stringify
        return json.dumps(str(obj), default=str)
    except Exception as ex:
        logger.debug("safe_serialize failed: %s", ex)
        try:
            return json.dumps(str(obj), default=str)
        except Exception:
            # Very last fallback
            return "\"<unserializable>\""


# -----------------------------------------------------------
# RAW FEEDBACK PUSHER
# -----------------------------------------------------------

def push_raw_feedback(results: List[Any]):
    """
    Pushes sentiment results with metadata.
    results list contains SentimentResult objects.
    """
    client = _client()
    if not client:
        return False

    sheet = client.open_by_key(GOOGLE_SHEET_ID)

    headers = [
        "timestamp",
        "id",
        "source",
        "text",
        "label",
        "score",
        "polarity",
        "toxicity",
        "emotions",
        "sarcasm",
        "aspects"
    ]

    ws = _get_or_create_ws(sheet, "raw_feedback", headers)

    rows = []
    ts = datetime.datetime.utcnow().isoformat()

    for r in results:
        # Safely serialize fields that might contain complex objects
        text_val = getattr(r, "text", "")
        if not isinstance(text_val, str):
            text_val = safe_serialize(text_val)
        text_val = text_val[:5000]  # respect length

        emotions_val = getattr(r, "emotions", {})
        emotions_serialized = safe_serialize(emotions_val)

        aspects_val = getattr(r, "aspects", {})
        aspects_serialized = safe_serialize(aspects_val)

        row = [
            ts,
            getattr(r, "id", ""),
            getattr(r, "source", ""),
            text_val,
            getattr(r, "label", ""),
            getattr(r, "score", ""),
            getattr(r, "polarity", ""),
            getattr(r, "toxicity", ""),
            emotions_serialized,
            getattr(r, "sarcasm", False),
            aspects_serialized
        ]

        rows.append(row)

    # Batch upload
    for i in range(0, len(rows), MAX_BATCH):
        chunk = rows[i:i + MAX_BATCH]
        ws.append_rows(chunk, value_input_option="USER_ENTERED")
        logger.info(f"Uploaded {len(chunk)} raw feedback rows.")

    return True


# -----------------------------------------------------------
# AGGREGATE METRICS PUSHER
# -----------------------------------------------------------

def push_aggregates(metrics: Dict[str, Any]):
    """
    Append KPI metrics such as:
    - avg_score, pos_ratio, neg_ratio, toxicity, emotion distribution
    """
    client = _client()
    if not client:
        return False

    sheet = client.open_by_key(GOOGLE_SHEET_ID)

    headers = [
        "timestamp", "total_records", "avg_score",
        "pos_count", "neg_count", "neu_count",
        "pct_positive", "pct_negative",
        "avg_toxicity", "dominant_emotion"
    ]

    ws = _get_or_create_ws(sheet, "aggregates", headers)

    ts = datetime.datetime.utcnow().isoformat()

    row = [
        ts,
        metrics.get("total", 0),
        metrics.get("avg_score", 0),
        metrics.get("pos_count", 0),
        metrics.get("neg_count", 0),
        metrics.get("neu_count", 0),
        metrics.get("pct_positive", 0),
        metrics.get("pct_negative", 0),
        metrics.get("avg_toxicity", 0),
        metrics.get("dominant_emotion", "unknown")
    ]

    ws.append_row(row)
    logger.info("Aggregates row added.")
    return True


# -----------------------------------------------------------
# A/B TEST RESULTS PUSHER
# -----------------------------------------------------------

def push_ab_test_results(campaign_id: str, results: List[Dict]):
    """
    Writes A/B test variants summary into Google Sheets.

    results = [
        {
            "variant": "variant_1",
            "impressions": 1500,
            "clicks": 123,
            "conversions": 12,
            "ctr": 0.082,
            "conv_rate": 0.097
        },
        ...
    ]
    """
    client = _client()
    if not client:
        return False

    sheet = client.open_by_key(GOOGLE_SHEET_ID)

    headers = [
        "timestamp",
        "campaign_id",
        "variant",
        "impressions",
        "clicks",
        "conversions",
        "ctr",
        "conv_rate"
    ]

    ws = _get_or_create_ws(sheet, "ab_test_results", headers)
    ts = datetime.datetime.utcnow().isoformat()

    rows = []
    for v in results:
        rows.append([
            ts,
            campaign_id,
            v.get("variant"),
            v.get("impressions", 0),
            v.get("clicks", 0),
            v.get("conversions", 0),
            v.get("ctr", 0),
            v.get("conv_rate", 0)
        ])

    for i in range(0, len(rows), MAX_BATCH):
        batch = rows[i:i + MAX_BATCH]
        ws.append_rows(batch)
        logger.info(f"Uploaded {len(batch)} A/B rows.")

    return True


# -----------------------------------------------------------
# CAMPAIGN LOGS PUSHER
# -----------------------------------------------------------

def log_campaign_event(event: str, info: Dict[str, Any]):
    """
    Stores high-level events:
        log_campaign_event("A/B test started", {campaign_name:"...", variants:3})
    """

    client = _client()
    if not client:
        return False

    sheet = client.open_by_key(GOOGLE_SHEET_ID)

    headers = ["timestamp", "event", "details"]

    ws = _get_or_create_ws(sheet, "campaign_logs", headers)

    ts = datetime.datetime.utcnow().isoformat()

    ws.append_row([ts, event, safe_serialize(info)])
    logger.info("Campaign event logged.")
    return True


# -----------------------------------------------------------
# Manual Test
# -----------------------------------------------------------

if __name__ == "__main__":
    print("\nTracker2 Test Running...\n")

    test_metrics = {
        "total": 120,
        "avg_score": 0.67,
        "pos_count": 60,
        "neg_count": 25,
        "neu_count": 35,
        "pct_positive": 0.50,
        "pct_negative": 0.20,
        "avg_toxicity": 0.12,
        "dominant_emotion": "joy",
    }

    push_aggregates(test_metrics)

    push_ab_test_results(
        "Campaign123",
        [
            {"variant": "v1", "impressions": 1000, "clicks": 80, "conversions": 10, "ctr": 0.08, "conv_rate": 0.125},
            {"variant": "v2", "impressions": 1000, "clicks": 65, "conversions": 8, "ctr": 0.065, "conv_rate": 0.123},
        ]
    )

    log_campaign_event("Demo event", {"info": "This is only a test."})
