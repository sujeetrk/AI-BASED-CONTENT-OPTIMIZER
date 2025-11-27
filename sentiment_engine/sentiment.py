"""
Improved sentiment.py

Major Enhancements:
-------------------
1. Multi-Model Sentiment Pipeline
   - HuggingFace transformer (primary)
   - VADER fallback (lightweight)
   - TextBlob for polarity (fine-grained score)
   - Safety classification for toxicity/hate/sarcasm

2. Emotion Classification (6-label model)
   - joy, sadness, anger, fear, love, surprise

3. Sarcasm Detection
   - Lightweight binary classifier (HuggingFace)

4. Aspect-Based Sentiment (ABSA-Lite)
   - Extracts key aspects/keywords and assigns local sentiment

5. Unified Output Schema:
   {
     text, label, score,
     emotions: {...},
     sarcasm: bool,
     toxicity: float,
     aspects: {...},
     polarity: float
   }

6. Batch Processing + Error Safe Fallbacks
"""

import os
import logging
from typing import List, Dict
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# -----------------------
# Config
# -----------------------
SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "hf")  # "hf" | "vader"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SARCASTIC_MODEL = "helinivan/sarcasm-detector-roberta"
TOXIC_MODEL = "unitary/toxic-bert"

# Lazy loaders
_hf_sentiment = None
_emotion_classifier = None
_sarcasm_classifier = None
_toxic_classifier = None
_vader = None

# -----------------------
# Result Schema
# -----------------------
@dataclass
class SentimentResult:
    text: str
    label: str
    score: float
    emotions: Dict = None
    sarcasm: bool = False
    toxicity: float = 0.0
    aspects: Dict = field(default=None, repr=False)   # <-- FIX HERE
    polarity: float = 0.0
    source: str = ""
    id: str = ""


# -----------------------
# Helper Initializers
# -----------------------

def _init_hf_sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis", truncation=True)

def _init_emotion():
    from transformers import pipeline
    return pipeline("text-classification", model=EMOTION_MODEL, top_k=None)

def _init_sarcasm():
    from transformers import pipeline
    return pipeline("text-classification", model=SARCASTIC_MODEL)

def _init_toxic():
    from transformers import pipeline
    return pipeline("text-classification", model=TOXIC_MODEL)

def _init_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()


# -----------------------
# Extra NLP Helpers
# -----------------------

from textblob import TextBlob


def extract_aspects(text: str, top_n: int = 3) -> Dict:
    """
    ABSA-lite:
    Extracts important nouns from text and assigns sentiment
    using local polarity around them.
    """
    blob = TextBlob(text)
    noun_phrases = blob.noun_phrases[:top_n]

    aspects = {}
    for np in noun_phrases:
        # simple polarity for the noun phrase
        aspects[np] = TextBlob(np).sentiment.polarity

    return aspects


# -----------------------
# Main Pipeline
# -----------------------

def analyze_texts(records: List[Dict]) -> List[SentimentResult]:
    """
    Full pipeline:
    1. HuggingFace model → main sentiment
    2. TextBlob → nuanced polarity score
    3. Emotion model → emotional fingerprint
    4. Sarcasm model → sarcasm flag
    5. Toxicity detection → toxicity score
    6. ABSA aspect extraction
    """

    global _hf_sentiment, _emotion_classifier, _sarcasm_classifier, _toxic_classifier

    texts = [r.get("text", "") for r in records]
    results: List[SentimentResult] = []

    # -------------------------------
    # Step 1: Init sentiment model
    # -------------------------------
    use_vader = False
    if SENTIMENT_BACKEND == "hf":
        try:
            if _hf_sentiment is None:
                _hf_sentiment = _init_hf_sentiment()
            hf_preds = _hf_sentiment(texts, truncation=True)
        except Exception as e:
            logger.warning("HF sentiment failed → using VADER. Error=%s", e)
            use_vader = True

    # -------------------------------
    # Step 2: Init optional models
    # -------------------------------
    if _emotion_classifier is None:
        try:
            _emotion_classifier = _init_emotion()
        except Exception:
            _emotion_classifier = None

    if _sarcasm_classifier is None:
        try:
            _sarcasm_classifier = _init_sarcasm()
        except Exception:
            _sarcasm_classifier = None

    if _toxic_classifier is None:
        try:
            _toxic_classifier = _init_toxic()
        except Exception:
            _toxic_classifier = None

    # -------------------------------
    # Step 3: Iterate records
    # -------------------------------
    for i, rec in enumerate(records):
        text = rec.get("text", "")

        # 1. Main Sentiment
        if not use_vader:
            pred = hf_preds[i]
            label_raw = pred["label"].upper()
            score_raw = float(pred["score"])

            if label_raw.startswith("NEG"):
                label = "NEGATIVE"
                score = 1 - score_raw
            elif label_raw.startswith("POS"):
                label = "POSITIVE"
                score = score_raw
            else:
                label = "NEUTRAL"
                score = 0.5
        else:
            # VADER fallback
            _init_vader()
            vs = _vader.polarity_scores(text)
            compound = vs["compound"]
            score = (compound + 1) / 2
            label = (
                "POSITIVE" if compound >= 0.05
                else "NEGATIVE" if compound <= -0.05
                else "NEUTRAL"
            )

        # 2. Polarity (TextBlob)
        polarity = TextBlob(text).sentiment.polarity

        # 3. Emotion Analysis
        emotions = {}
        if _emotion_classifier:
            try:
                emo_pred = _emotion_classifier(text)[0]
                # Convert list of dicts to simple map
                emotions = {e["label"]: float(e["score"]) for e in emo_pred}
            except Exception:
                emotions = {}

        # 4. Sarcasm
        sarcasm_flag = False
        if _sarcasm_classifier:
            try:
                sar_pred = _sarcasm_classifier(text)[0]
                sarcasm_flag = True if sar_pred["label"].lower() == "sarcastic" else False
            except Exception:
                sarcasm_flag = False

        # 5. Toxicity Score
        toxicity_score = 0.0
        if _toxic_classifier:
            try:
                tx = _toxic_classifier(text)[0]["score"]
                toxicity_score = float(tx)
            except Exception:
                toxicity_score = 0.0

        # 6. Aspect Extraction (ABSA-lite)
        aspects = extract_aspects(text)

        results.append(
            SentimentResult(
                text=text,
                label=label,
                score=round(score, 4),
                emotions=emotions,
                sarcasm=sarcasm_flag,
                toxicity=float(toxicity_score),
                aspects=aspects,
                polarity=polarity,
                source=rec.get("source", ""),
                id=rec.get("id", "")
            )
        )

    return results


# -----------------------
# Metrics Aggregator
# -----------------------

def aggregate_metrics(results: List[SentimentResult]) -> Dict:
    """
    Aggregates:
    - avg sentiment score
    - emotion distribution
    - top negative texts
    - toxicity average
    """
    import numpy as np

    scores = [r.score for r in results]
    avg_score = np.mean(scores) if scores else 0.0

    pos = sum(1 for r in results if r.label == "POSITIVE")
    neg = sum(1 for r in results if r.label == "NEGATIVE")
    neu = sum(1 for r in results if r.label == "NEUTRAL")
    total = len(results)

    # Emotion summary
    emotion_totals = {}
    for r in results:
        for e, val in (r.emotions or {}).items():
            emotion_totals[e] = emotion_totals.get(e, 0) + val

    # Toxicity average
    avg_toxicity = np.mean([r.toxicity for r in results]) if results else 0.0

    # Worst sentiment examples
    worst = sorted(results, key=lambda x: x.score)[:5]
    top_negative = [{"text": r.text, "score": r.score, "id": r.id} for r in worst]

    return {
        "total": total,
        "avg_score": round(avg_score, 4),
        "pos_count": pos,
        "neg_count": neg,
        "neu_count": neu,
        "pct_positive": round(pos/total, 4) if total else 0,
        "pct_negative": round(neg/total, 4) if total else 0,
        "avg_toxicity": round(avg_toxicity, 4),
        "emotion_distribution": emotion_totals,
        "top_negative": top_negative,
    }


# -----------------------
# Test
# -----------------------

if __name__ == "__main__":
    sample = [
        {"id": "1", "text": "I absolutely love this AI tool! It's mind-blowing.", "source": "twitter"},
        {"id": "2", "text": "This product is terrible... I hate the experience.", "source": "instagram"},
        {"id": "3", "text": "Well, that was *great*... sure 🙄", "source": "youtube"},
    ]

    out = analyze_texts(sample)
    for r in out:
        print("\nResult:", r)

    print("\nAggregated Metrics:")
    print(aggregate_metrics(out))
