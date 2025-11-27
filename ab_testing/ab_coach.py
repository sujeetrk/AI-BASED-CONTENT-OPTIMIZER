"""
Improved ab_coach2.py

Purpose:
--------
This module now provides a COMPLETE A/B testing framework with:

1. Content variant generation (via content_engine)
2. Sentiment scoring + trend score integration
3. Statistical significance testing (Z-test)
4. Optional Bayesian A/B testing
5. ML-based success probability prediction
6. Logging into metrics hub + Slack winner notification
7. Clean return structure for APIs and dashboards

Designed to match the project PDF requirements (Milestone 4).
"""

import uuid
import numpy as np
import pandas as pd
import os
import joblib
from typing import Dict, List, Optional

# ---------------------------
# Imports from your project
# ---------------------------
from app.content_engine.content_generator3 import generate_final_variations
from app.sentiment_engine.sentiment2 import analyze_texts
from app.content_engine.trend_based_optimizer3 import optimize_content
from app.metrics_engine.metrics_hub2 import record_campaign_metrics, fetch_campaign_history
from app.notifications.slack_notifier2 import send_ab_test_winner


MODEL_PATH = os.getenv("MODEL_PATH", "models/predictor.joblib")


# ------------------------------------------------------------
# Utility: Frequentist Z-test for proportions
# ------------------------------------------------------------
def z_test_conversion_rate(clicks_A, conv_A, clicks_B, conv_B):
    """
    Performs a two-sample Z-test for comparing conversion rates.
    Returns:
        z_score, p_value
    """
    if clicks_A == 0 or clicks_B == 0:
        return 0, 1  # cannot compute Z-score

    p1 = conv_A / clicks_A
    p2 = conv_B / clicks_B

    p_pool = (conv_A + conv_B) / (clicks_A + clicks_B)
    denominator = np.sqrt(p_pool * (1 - p_pool) * (1/clicks_A + 1/clicks_B))

    if denominator == 0:
        return 0, 1

    z = (p1 - p2) / denominator

    # Two-tailed test p-value
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value


# ------------------------------------------------------------
# Utility: Bayesian A/B Testing (Beta-Bernoulli)
# ------------------------------------------------------------
def bayesian_probability_winner(clicks, conversions):
    """
    Returns posterior probability that each variant is better.
    Prior: Beta(1,1)
    """
    variants = list(clicks.keys())
    samples = 20000

    posterior_samples = {}
    for v in variants:
        posterior_samples[v] = np.random.beta(1 + conversions[v], 
                                              1 + clicks[v] - conversions[v], 
                                              size=samples)

    # Probability each variant is max
    probs = {}
    for v in variants:
        probs[v] = float(np.mean(posterior_samples[v] == np.max(list(posterior_samples[x] for x in variants))))

    return probs


# ------------------------------------------------------------
# A/B Test Generator
# ------------------------------------------------------------
def run_ab_test(
    campaign_name: str,
    topic: str,
    platform: str,
    keywords: List[str],
    audience: str,
    tone: str = "positive",
    n_variants: int = 3,
    impressions_per_variant: int = 1500,
    existing_variations: Optional[List[Dict]] = None
) -> Dict:
    """
    Runs a full A/B test:
    1. Generate content variants if not generated
    2. Score sentiment + trend
    3. Simulate impressions/clicks/conversions (or real data later)
    4. Compute stats significance
    5. Send Slack winner notification
    6. Store metrics in metrics hub
    """

    campaign_id = f"{campaign_name}-{uuid.uuid4().hex[:6]}"

    # 1. Generate content variants
    if existing_variations is not None:
        variations = existing_variations
    else:
        variations = generate_final_variations(
            topic=topic,
            platform=platform,
            keywords=keywords,
            audience=audience,
            tone=tone,
            n=n_variants
        )

    summary = []
    click_data = {}
    conv_data = {}

    for idx, var in enumerate(variations, start=1):
        text = var["text"]

        # 2. Sentiment & Trend scoring
        sent_result = analyze_texts([
            {"id": "1", "text": text, "source": "ab_test"}])[0]

        sent = {
            "polarity": sent_result.polarity,
            "label": sent_result.label,
            "score": sent_result.score,
            "emotions": sent_result.emotions,
            "toxic": sent_result.toxicity,
            "aspects": sent_result.aspects,
            "viral_potential": float(sent_result.score)  # use sentiment score as viral_potential
        }
        trend_info = optimize_content(text)

        # 3. Simulated metrics (can be replaced with real metrics)
        impressions = impressions_per_variant

        base_ctr = 0.02 + sent["viral_potential"] * 0.1 + trend_info["final_sentiment"] * 0.05
        clicks = np.random.binomial(impressions, min(0.5, base_ctr))

        base_conv = 0.01 + sent["polarity"] * 0.03 + trend_info["final_sentiment"] * 0.02
        conv_rate = max(0.001, min(0.5, base_conv))
        conversions = np.random.binomial(clicks, conv_rate)

        ctr = clicks / impressions if impressions > 0 else 0
        conversion_rate = conversions / clicks if clicks > 0 else 0

        # store for Bayesian computation
        click_data[f"variant_{idx}"] = int(clicks)
        conv_data[f"variant_{idx}"] = int(conversions)

        # 4. Save metrics
        record_campaign_metrics(
            campaign_id=campaign_id,
            variant=f"variant_{idx}",
            impressions=impressions,
            clicks=int(clicks),
            conversions=int(conversions),
            sentiment_score=float(sent["polarity"]),
            trend_score=float(trend_info["final_sentiment"])
        )

        summary.append({
            "variant": f"variant_{idx}",
            "text": text,
            "impressions": impressions,
            "clicks": int(clicks),
            "conversions": int(conversions),
            "ctr": ctr,
            "conv_rate": conversion_rate,
            "sentiment": sent,
            "trend": trend_info
        })

    # Convert to dataframe
    df = pd.DataFrame(summary)
    winner = df.loc[df["conv_rate"].idxmax()].to_dict()

    # ---------------------------------------------------------
    # Frequentist significance test (best vs second best)
    # ---------------------------------------------------------
    df_sorted = df.sort_values("conv_rate", ascending=False)
    if len(df_sorted) > 1:
        A = df_sorted.iloc[0]
        B = df_sorted.iloc[1]
        z, p_val = z_test_conversion_rate(
            A["clicks"], A["conversions"],
            B["clicks"], B["conversions"]
        )
    else:
        z, p_val = 0, 1

    # ---------------------------------------------------------
    # Bayesian Winner Probability
    # ---------------------------------------------------------
    bayes_probs = bayesian_probability_winner(click_data, conv_data)

    # Slack notification
    send_ab_test_winner(campaign_id, winner)

    return {
        "campaign_id": campaign_id,
        "summary": df.to_dict(orient="records"),
        "winner": winner,
        "significance": {
            "z_score": z,
            "p_value": p_val
        },
        "bayesian_probabilities": bayes_probs
    }


# ------------------------------------------------------------
# ML Recommendation System
# ------------------------------------------------------------
def recommend(features: Dict) -> Dict:
    """
    Uses trained ML model to predict success probability.
    If model missing → uses heuristic.
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        df = pd.DataFrame([features])

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]
        else:
            prob = model.predict(df)[0]

        return {"predicted_success_prob": float(prob)}

    # Fallback heuristic
    score = features.get("avg_sentiment", 0.0) * 0.6 + features.get("historical_ctr", 0.01) * 0.4
    return {"predicted_success_prob": round(max(0, min(1, score)), 4)}


# ------------------------------------------------------------
# Local Debug
# ------------------------------------------------------------
if __name__ == "__main__":
    result = run_ab_test(
        campaign_name="summer_sale",
        topic="AI-powered sunglasses",
        platform="Twitter",
        keywords=["#AI", "#Summer"],
        audience="tech enthusiasts",
        tone="positive",
        n_variants=3
    )

    print("\nA/B Test Result:")
    print(result)
