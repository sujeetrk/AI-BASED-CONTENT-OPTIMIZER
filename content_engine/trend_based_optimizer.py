import random
from textblob import TextBlob

# -------------------------------------------------------------------
# TREND DATA (Mocked – connect real API later)
# -------------------------------------------------------------------
TREND_DATA = {
    "AI": 0.79,
    "Automation": 0.51,
    "GenAI": 0.51,
    "DigitalMarketing": 0.39,
    "BlackFriday": 0.34625
}

# -------------------------------------------------------------------
# TREND SCORE HELPERS
# -------------------------------------------------------------------

def hashtag_trend_score(tag: str) -> float:
    """
    Return trend score for a hashtag (case-insensitive).
    """
    tag_clean = tag.replace("#", "").lower()
    for k, v in TREND_DATA.items():
        if k.lower() == tag_clean:
            return v
    return 0.0


# -------------------------------------------------------------------
# HASHTAG CLEANUP (dedupe + move to end + trend sort)
# -------------------------------------------------------------------

def dedupe_hashtags_in_text(text: str) -> str:
    parts = text.split()
    seen = set()
    result = []

    for p in parts:
        if p.startswith("#"):
            if p.lower() not in seen:
                result.append(p)
                seen.add(p.lower())
        else:
            result.append(p)

    return " ".join(result)


def move_hashtags_to_end(text: str) -> str:
    """
    Extract hashtags, dedupe, sort by trend score, append at end.
    """
    words = text.split()
    hashtags = []
    normal_words = []
    seen = set()

    for w in words:
        if w.startswith("#"):
            wl = w.lower()
            if wl not in seen:
                hashtags.append(w)
                seen.add(wl)
        else:
            normal_words.append(w)

    if not hashtags:
        return text

    hashtags = sorted(hashtags, key=lambda h: hashtag_trend_score(h), reverse=True)

    final_text = " ".join(normal_words).strip()
    final_tags = " ".join(hashtags)

    return f"{final_text} {final_tags}".strip()


# -------------------------------------------------------------------
# TREND FETCHING
# -------------------------------------------------------------------

def fetch_trends(n: int = 5):
    return sorted(
        [f"#{k}" for k in TREND_DATA.keys()],
        key=lambda x: TREND_DATA[x[1:]],
        reverse=True
    )[:n]


def smart_inject_hashtags(text: str, hashtags: list):
    return (text + " " + " ".join(hashtags[:3])).strip()


# -------------------------------------------------------------------
# MAIN OPTIMIZER
# -------------------------------------------------------------------

def optimize_content(text: str, keyword: str = "marketing"):
    trends = fetch_trends(5)

    trend_scores = {f"#{k}": v for k, v in TREND_DATA.items()}

    optimized = text

    optimized = dedupe_hashtags_in_text(optimized)
    optimized = move_hashtags_to_end(optimized)

    raw_sent = {
        k.lower(): 0.0  # your per-tag sentiment logic can go here
        for k in trend_scores.keys()
    }

    final_sentiment = sum(raw_sent.values()) / len(raw_sent)

    return {
        "optimized_text": optimized,
        "applied_hashtags": list(dict.fromkeys(trends)),
        "trend_scores": trend_scores,
        "raw_sentiments": raw_sent,
        "final_sentiment": final_sentiment
    }
