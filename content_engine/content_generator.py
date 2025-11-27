# app/content_engine/content_generator3.py

import genai
from typing import List, Dict, Optional
import os
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# Attempt to import Groq client if present
try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False
    GEMINI_AVAILABLE = True


from .dynamic_prompt2 import generate_engaging_prompt

# USING OPTIMIZER 3 (correct one)
from app.content_engine.trend_based_optimizer3 import (
    optimize_content,
    hashtag_trend_score,
    dedupe_hashtags_in_text,
    move_hashtags_to_end
)

# Optional quality tools (may not be installed in all environments)
try:
    import textstat  # readability metrics
    TEXTSTAT_AVAILABLE = True
except Exception:
    TEXTSTAT_AVAILABLE = False

try:
    import language_tool_python  # grammar check
    LT_AVAILABLE = True
except Exception:
    LT_AVAILABLE = False


# ---------------------------
# Hashtag Cleaning (Final Logic)
# ---------------------------

def clean_punctuation_hashtags(text: str) -> str:
    """
    Remove trailing punctuation from hashtags (#AI, -> #AI).
    """
    words = text.split()
    cleaned = []
    for w in words:
        if w.startswith("#"):
            w = w.rstrip(",.?!;:")  # remove comma, period, semicolon, etc.
        cleaned.append(w)
    return " ".join(cleaned)

def clean_and_order_hashtags(text: str):
    """
    1. Remove duplicates
    2. Sort hashtags by trend score
    3. Move hashtags to end
    """
    # Step 0: clean raw punctuation (#AI, -> #AI)
    text = clean_punctuation_hashtags(text)

    # Step 1: dedupe
    cleaned = dedupe_hashtags_in_text(text)

    # Step 2: move & sort using trend score
    cleaned = move_hashtags_to_end(cleaned)

    return cleaned


# ---------------------------
# Local Fallback Generator
# ---------------------------

def _local_generate(prompt: str, n: int = 3) -> List[str]:
    return [f"{prompt} — variant {i+1}" for i in range(n)]


# ---------------------------
# Trend Fetching (Stub)
# ---------------------------

def fetch_trends(platform: str = "twitter", n: int = 5) -> List[str]:
    """
    Static fallback trends.
    """
    sample = ["#AI", "#Marketing", "#GenAI", "#Growth", "#Automation"]
    return sample[:n]


# ---------------------------
# Groq Call Helper
# ---------------------------

def _call_groq(prompt: str, model: str = None) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    )
    return resp.choices[0].message.content

def _call_gemini(prompt: str, model: str = None) -> str:
    """
    Calls Gemini API as fallback if Groq fails.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    gmodel = genai.GenerativeModel(model)
    response = gmodel.generate_content(prompt)

    return response.text if hasattr(response, "text") else str(response)


def generate_variations(prompt: str, n: int = 3) -> List[str]:
    """
    Generate n variations with priority:
    1. Groq
    2. Gemini (fallback)
    3. Local fallback
    """

    # Try Groq
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        try:
            logger.info("Using Groq for generation...")
            return [_call_groq(prompt) for _ in range(n)]
        except Exception as e:
            logger.exception("Groq failed → Falling back to Gemini...")

    # Try Gemini
    if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        try:
            logger.info("Using Gemini fallback for generation...")
            return [_call_gemini(prompt) for _ in range(n)]
        except Exception as e:
            logger.exception("Gemini failed → Falling back to local generator...")

    # Final fallback
    logger.warning("Using LOCAL fallback generator (prompt echo).")
    return _local_generate(prompt, n)


# ---------------------------
# Quality Scoring Layer
# ---------------------------

def score_quality(text: str) -> Dict:
    readability_score = None
    grammar_issues = None

    if TEXTSTAT_AVAILABLE:
        try:
            readability_score = textstat.flesch_reading_ease(text)
        except:
            readability_score = None

    if LT_AVAILABLE:
        try:
            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(text)
            grammar_issues = len(matches)
        except:
            grammar_issues = None

    return {
        "readability_score": readability_score,
        "grammar_issues": grammar_issues
    }


# ---------------------------
# Engagement-Aware Ranking
# ---------------------------

def optimize_with_engagement(candidates: List[Dict], past_metrics: Optional[Dict] = None) -> List[Dict]:

    top_keywords = []
    if past_metrics:
        top_keywords = list(past_metrics.get("top_keywords", []))[:3]

    scored = []

    for c in candidates:
        text = c.get("optimized_text", "")
        score = 0.0

        q = score_quality(text)

        if q.get("readability_score") is not None:
            score += (q["readability_score"] / 100.0)

        if q.get("grammar_issues") is not None:
            score -= min(1.0, 0.1 * q["grammar_issues"])

        for kw in top_keywords:
            if kw.lower() in text.lower():
                score += 0.2

        for t in fetch_trends():
            if t.lower() in text.lower():
                score += 0.15

        c["engagement_score"] = score
        scored.append((score, c))

    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)

    return [c for _, c in scored_sorted]


# ---------------------------
# FINAL PIPELINE
# ---------------------------

def generate_final_variations(
    topic: str,
    platform: str,
    keywords: List[str],
    audience: str,
    tone: str = "positive",
    n: int = 3,
    word_count: int = 50,
    past_metrics: Optional[Dict] = None
) -> List[Dict]:

    if isinstance(keywords, str):
        keywords = keywords.split(",")

    trends = fetch_trends(platform, n=5)

    injected_keywords = keywords + [t for t in trends if t not in keywords]

    prompt = generate_engaging_prompt(
        topic,
        platform,
        injected_keywords,
        audience,
        tone,
        trends=trends,
        word_count=word_count
    )

    # Generate raw variations
    candidates = generate_variations(prompt, n=n)

    # Apply Trend-Based Optimizer 3 + hashtag cleaning
    optimized_candidates = []
    for c in candidates:
        opt = optimize_content(c)

        cleaned = clean_and_order_hashtags(opt["optimized_text"])
        opt["optimized_text"] = cleaned

        optimized_candidates.append(opt)

    # Engagement Ranking
    final_order = optimize_with_engagement(optimized_candidates, past_metrics)

    results = []
    for item in final_order:
        optimized_text = item.get("optimized_text", "")

        results.append({
            "text": optimized_text,
            "quality": score_quality(optimized_text),
            "meta": {
                "topic": topic,
                "platform": platform,
                "audience": audience,
                "injected_keywords": injected_keywords,
                "applied_hashtags": item.get("applied_hashtags", []),
                "trend_scores": item.get("trend_scores", {}),
                "final_sentiment": item.get("final_sentiment")
            }
        })

    return results


# ---------------------------
# Test Run
# ---------------------------

if __name__ == "__main__":
    out = generate_final_variations(
        "AI in Marketing",
        "Twitter",
        ["#AI", "#Marketing"],
        "marketers",
        "positive",
        n=3
    )

    for i, r in enumerate(out, 1):
        print(f"\nVariant {i}:")
        print("Text:", r["text"])
        print("Quality:", r["quality"])
        print("Meta:", r["meta"])
