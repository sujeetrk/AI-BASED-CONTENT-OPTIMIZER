"""
train_model3.py

Purpose:
--------
Trains a predictive model that estimates the probability that a content variant
(campaign version) will succeed in future A/B tests.

Model Input Features:
---------------------
- ctr
- sentiment
- polarity
- conversions
- trend_score

Label:
------
success = 1 if conversion_rate > 0.02 else 0

Output:
-------
models/predictor_TIMESTAMP.joblib
models/predictor.joblib
"""

import os
import joblib
import pandas as pd
import numpy as np
import datetime
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE

# Load CSV through this function
from app.metrics_engine.metrics_hub3 import get_ml_training_data


# ---------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    logger.addHandler(h)

# ---------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Latest model
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "predictor.joblib")


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def compute_success_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    success = 1 when conversion_rate > threshold
    threshold = 2%
    """
    df = df.copy()

    df["conversion_rate"] = df["conversions"] / df["ctr"].replace(0, np.nan)
    df["conversion_rate"] = df["conversion_rate"].fillna(0)

    df["success"] = (df["conversion_rate"] > 0.02).astype(int)
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering:
    - normalize CTR
    - combine sentiment + polarity
    - trend_score smoothing
    """

    df = df.copy()

    df["ctr_norm"] = df["ctr"].clip(0, 1)
    df["sentiment_norm"] = df["sentiment"].clip(-1, 1)
    df["polarity_norm"] = df["polarity"].clip(-1, 1)
    df["trend_norm"] = df["trend_score"].clip(-1, 1)

    features = df[[
        "ctr_norm",
        "sentiment_norm",
        "polarity_norm",
        "trend_norm",
        "conversions"
    ]]

    return features


# ---------------------------------------------------------------
# Train Model
# ---------------------------------------------------------------

def train():
    logger.info("Loading ML training dataset...")
    df = get_ml_training_data()

    if df.empty:
        raise ValueError("Training dataset is empty. Collect more A/B campaign data.")

    # Step 1: Add success label
    df = compute_success_label(df)

    # Step 2: Feature Engineering
    X = feature_engineer(df)
    y = df["success"]

    logger.info(f"Dataset size: {len(df)} rows")
    logger.info(f"Class balance:\n{df['success'].value_counts()}")

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Step 4: Balance classes with SMOTE
    sm = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

    logger.info("After SMOTE balancing:")
    logger.info(y_train_balanced.value_counts())

    # Step 5: Base model
    base_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=None
    )

    # Step 6: Hyperparameter tuning
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [None, 8, 12],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        base_model,
        param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=3,
        verbose=1
    )

    logger.info("Running GridSearchCV...")
    grid.fit(X_train_balanced, y_train_balanced)

    best_model = grid.best_estimator_
    logger.info(f"Best params: {grid.best_params_}")

    # Step 7: Evaluate
    preds = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    pred_labels = best_model.predict(X_test)

    f1 = f1_score(y_test, pred_labels)
    precision = precision_score(y_test, pred_labels)
    recall = recall_score(y_test, pred_labels)
    cm = confusion_matrix(y_test, pred_labels)

    logger.info(f"Model AUC: {auc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Step 8: Save Model (with timestamp)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = os.path.join(MODEL_DIR, f"predictor_{timestamp}.joblib")

    joblib.dump(best_model, versioned_path)
    joblib.dump(best_model, LATEST_MODEL_PATH)

    logger.info(f"Model saved: {versioned_path}")
    logger.info(f"Latest model updated: {LATEST_MODEL_PATH}")

    return {
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "model_path": versioned_path,
        "best_params": grid.best_params_
    }


# ---------------------------------------------------------------
# Run manually
# ---------------------------------------------------------------

if __name__ == "__main__":
    results = train()
    print("\nTraining Results:")
    print(results)
