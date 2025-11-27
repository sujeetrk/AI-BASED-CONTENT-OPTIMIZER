"""
Improved sentiment_analyzer2.py

Purpose:
--------
A lightweight alternative sentiment engine used for:
- Fast batch analysis
- Quick preprocessing before deep analysis
- Low-resource environments (e.g., student laptops)

Enhancements:
-------------
1. Multi-task classifier:
   - Sentiment (POS/NEG/NEU)
   - Emotion (joy, sadness, anger, fear, love, surprise)

2. Batch support
3. Multi-language support (auto-detection using langdetect)
4. Normalized confidence output
5. Simple, predictable output format compatible with the main sentiment pipeline
6. Error-safe fallbacks (TextBlob if HF unavailable)
"""

import logging
from typing import List, Union, Dict

from textblob import TextBlob
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    from langdetect import detect
    LANG_AVAILABLE = True
except Exception:
    LANG_AVAILABLE = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------------------------------------
# 1. Lazy-loaded Models
# ----------------------------------------------------

_senti_model = None
_emotion_model = None


def _init_sentiment_model():
    """
    Lightweight model for sentiment classification.
    """
    return pipeline("sentiment-analysis")


def _init_emotion_model():
    """
    Multi-label emotion classification model.
    """
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )


# ----------------------------------------------------
# 2. Helper Functions
# ----------------------------------------------------

def detect_language(text: str) -> str:
    if not LANG_AVAILABLE:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def fallback_sentiment(text: str) -> Dict:
    """
    Fallback using TextBlob polarity (-1..1)
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity >= 0.05:
        label = "POSITIVE"
    elif polarity <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "score": abs(polarity),
        "polarity": polarity
    }


def simplify_emotion_output(raw_output: List[Dict]) -> Dict:
    return {x["label"]: float(x["score"]) for x in raw_output}


# ----------------------------------------------------
# 3. Main Analyzer
# ----------------------------------------------------

def analyze_sentiment(texts: Union[str, List[str]]) -> List[Dict]:
    """
    Core lightweight sentiment + emotion analyzer.
    """
    if isinstance(texts, str):
        texts = [texts]

    global _senti_model, _emotion_model

    results = []

    # Load models if available
    if HF_AVAILABLE:
        if _senti_model is None:
            _senti_model = _init_sentiment_model()
        if _emotion_model is None:
            _emotion_model = _init_emotion_model()

    for text in texts:

        # --------------------------------
        # Language Detection
        # --------------------------------
        lang = detect_language(text)

        # --------------------------------
        # 1. Sentiment
        # --------------------------------
        if HF_AVAILABLE:
            try:
                hf_pred = _senti_model(text)[0]
                label = hf_pred["label"].upper()
                score = float(hf_pred["score"])
                polarity = TextBlob(text).sentiment.polarity
            except Exception:
                logger.warning("HF sentiment failed, using fallback")
                s = fallback_sentiment(text)
                label, score, polarity = s["label"], s["score"], s["polarity"]
        else:
            s = fallback_sentiment(text)
            label, score, polarity = s["label"], s["score"], s["polarity"]

        # Normalize NEGATIVE → return low score instead of high negativity
        if label.startswith("NEG"):
            norm_score = 1 - score
        else:
            norm_score = score

        # --------------------------------
        # 2. Emotion detection
        # --------------------------------
        emotions = {}
        if HF_AVAILABLE:
            try:
                emo_raw = _emotion_model(text)[0]
                emotions = simplify_emotion_output(emo_raw)
            except Exception:
                emotions = {}

        # --------------------------------
        # 3. Build output entry
        # --------------------------------
        results.append({
            "text": text,
            "sentiment_label": label,
            "sentiment_score": round(norm_score, 4),
            "polarity": polarity,
            "emotions": emotions,
            "language": lang
        })

    return results


# ----------------------------------------------------
# 4. DataFrame Helper
# ----------------------------------------------------

def analyze_from_dataframe(df, text_column: str):
    """
    Adds sentiment + emotion columns to a DataFrame.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    results = analyze_sentiment(df[text_column].tolist())

    df["sentiment_label"] = [r["sentiment_label"] for r in results]
    df["sentiment_score"] = [r["sentiment_score"] for r in results]
    df["polarity"] = [r["polarity"] for r in results]
    df["emotions"] = [r["emotions"] for r in results]
    df["language"] = [r["language"] for r in results]

    return df


# ----------------------------------------------------
# 5. Test Run
# ----------------------------------------------------

if __name__ == "__main__":
    sample_texts = [
        "I absolutely love this AI tool!",
        "This is frustrating and disappointing.",
        "I'm not sure if this is good or bad 😂",
    ]

    out = analyze_sentiment(sample_texts)
    for r in out:
        print("\n", r)
