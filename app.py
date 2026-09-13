import streamlit as st
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(
    page_title="Institutional Financial NLP Sentiment Benchmark",
    page_icon="🏦",
    layout="wide",
)

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]


@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def fetch_ticker_news(ticker, max_articles=8):
    """Fetch recent financial headlines for a ticker."""
    ticker = ticker.strip().upper()
    headlines = []

    # Yahoo Finance chart/news endpoint used through its public API response.
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get("news", [])[:max_articles]:
            title = item.get("title")
            if title:
                headlines.append(title)
    except Exception:
        pass

    # Fallback to Google News RSS.
    if not headlines:
        try:
            url = (
                "https://news.google.com/rss/search"
                f"?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            )
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            root = ET.fromstring(response.content)
            for item in root.findall(".//item")[:max_articles]:
                title_node = item.find("title")
                if title_node is not None and title_node.text:
                    title = title_node.text
                    if " - " in title:
                        title = title.rsplit(" - ", 1)[0]
                    headlines.append(title)
        except Exception:
            pass

    return list(dict.fromkeys(headlines))[:max_articles]


def analyze_sentiment_batch(records):
    """Apply the same FinBERT scoring logic as the notebook."""
    if not records:
        return pd.DataFrame()

    tokenizer, model, device = load_finbert()

    df = pd.DataFrame(records, columns=["ticker", "title"])

    encoded = tokenizer(
        df["title"].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pred_labels = []
    net_scores = []

    for p in probs.cpu():
        pos, neg, neu = p[0].item(), p[1].item(), p[2].item()
        pred_labels.append(LABELS[p.argmax().item()])
        net_scores.append(pos - neg)

    df["sentiment"] = pred_labels
    df["net_score"] = net_scores

    return df


def run_sector_benchmark(tickers_text):
    """Notebook's run_sector_benchmark adapted for Streamlit."""
    tickers = [
        x.strip().upper()
        for x in tickers_text.split(",")
        if x.strip()
    ]
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        return None, None, None, "Enter at least one ticker."

    if len(tickers) > 12:
        return None, None, None, "Please use no more than 12 tickers."

    records = []

    for ticker in tickers:
        for headline in fetch_ticker_news(ticker, max_articles=8):
            records.append((ticker, headline))

    if not records:
        return (
            None,
            None,
            None,
            "No recent headlines could be retrieved. Check the ticker symbols and try again.",
        )

    df_scored = analyze_sentiment_batch(records)

    if df_scored.empty:
        return None, None, None, "No headlines were available for sentiment analysis."

    leaderboard = (
        df_scored.groupby("ticker")
        .agg(
            Avg_Sentiment=("net_score", "mean"),
            Articles_Count=("title", "count"),
            Bullish_Pct=(
                "sentiment",
                lambda x: (x == "positive").mean() * 100,
            ),
            Bearish_Pct=(
                "sentiment",
                lambda x: (x == "negative").mean() * 100,
            ),
            Neutral_Pct=(
                "sentiment",
                lambda x: (x == "neutral").mean() * 100,
            ),
        )
        .reset_index()
        .sort_values("Avg_Sentiment", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(leaderboard) * 0.7))
    )
    ax.barh(
        leaderboard["ticker"],
        leaderboard["Avg_Sentiment"],
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Net Sentiment Score")
    ax.set_ylabel("Ticker")
    ax.set_title("Sector Cross-Sectional Sentiment Ranking")
    plt.tight_layout()

    top = leaderboard.iloc[0]
    bottom = leaderboard.iloc[-1]

    summary = f"""
### 📊 Executive Summary

- **Top Bullish Ticker:** `{top['ticker']}` — **{top['Avg_Sentiment']:.3f}**
- **Top Bearish Ticker:** `{bottom['ticker']}` — **{bottom['Avg_Sentiment']:.3f}**
- **Total Articles Analyzed:** **{len(df_scored)}**
- **Companies Compared:** **{len(leaderboard)}**
"""

    return summary, leaderboard, fig, df_scored, None


st.title("🏦 Institutional Financial NLP Sentiment Benchmark")
st.markdown(
    "Analyze and rank real-time market sentiment across competing equities "
    "using **ProsusAI FinBERT**."
)

st.info(
    "This Streamlit app is a deployment version of the notebook pipeline: "
    "news collection → FinBERT sentiment → net directional score → sector ranking."
)

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.write("Enter comma-separated stock tickers.")
    st.caption("Example: V, MA, AXP, PYPL")
    st.caption("Maximum 12 tickers.")

tickers = st.text_input(
    "Enter Comma-Separated Tickers",
    value="V, MA, AXP, PYPL",
    placeholder="e.g. NVDA, AMD, INTC, TSM",
)

if st.button("🔍 Analyze Market Sentiment", type="primary", width="stretch"):
    with st.spinner(
        "Fetching headlines and running FinBERT sentiment analysis..."
    ):
        result = run_sector_benchmark(tickers)

    if result[-1] is not None:
        st.error(result[-1])
    else:
        summary, leaderboard, fig, scored, _ = result

        st.markdown(summary)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Companies", len(leaderboard))
        c2.metric("Articles", len(scored))
        c3.metric(
            "Most Bullish",
            leaderboard.iloc[0]["ticker"],
            f"{leaderboard.iloc[0]['Avg_Sentiment']:+.3f}",
        )
        c4.metric(
            "Most Bearish",
            leaderboard.iloc[-1]["ticker"],
            f"{leaderboard.iloc[-1]['Avg_Sentiment']:+.3f}",
        )

        st.divider()

        st.subheader("📋 Sector Leaderboard Metrics")
        st.dataframe(
            leaderboard.round(3),
            width="stretch",
            hide_index=True,
        )

        st.subheader("📈 Relative Tone Comparison Chart")
        st.pyplot(fig)

        st.subheader("📰 Headlines Analyzed")
        st.dataframe(
            scored,
            width="stretch",
            hide_index=True,
        )

        csv = scored.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Sentiment Results",
            csv,
            "financial_sentiment_results.csv",
            "text/csv",
        )

        st.caption(
            "Net Directional Score = P(Positive) − P(Negative), "
            "as implemented in the original notebook."
        )

else:
    st.subheader("How the model works")
    a, b, c = st.columns(3)

    with a:
        st.markdown("**1. 📰 News Collection**\n\nRetrieve recent headlines for each ticker.")

    with b:
        st.markdown("**2. 🤖 FinBERT**\n\nClassify each headline as positive, negative, or neutral.")

    with c:
        st.markdown("**3. 📊 Benchmarking**\n\nRank companies by average net sentiment.")

    st.divider()
    st.markdown(
        "**Model:** `ProsusAI/finbert`  \n"
        "**Score:** `P(Positive) − P(Negative)`  \n"
        "**Framework:** PyTorch + Hugging Face Transformers"
    )
