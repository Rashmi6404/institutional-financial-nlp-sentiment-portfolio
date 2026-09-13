# 🏦 Institutional Financial NLP Sentiment Benchmark

🚀 **[Live Demo – Streamlit App] ((https://institutional-financial-nlp-sentiment-portfolio-2n2uh6xbgpvvgg.streamlit.app/))**

Financial news sentiment analysis and equity benchmarking using FinBERT, PyTorch, and Streamlit.

## 🚀 Streamlit App

The deployed application converts the notebook's Gradio interface into a Streamlit dashboard.

### Pipeline

**Financial News → FinBERT → Positive / Negative / Neutral → Net Sentiment → Sector Ranking**

## Features

- Recent financial-news headline collection
- FinBERT sentiment classification
- Positive / negative / neutral predictions
- Net directional sentiment score
- Multi-stock comparison
- Sector leaderboard
- Relative tone chart
- Downloadable sentiment results

## Model

**ProsusAI/finbert**

The notebook's scoring logic is preserved:

`Net Directional Score = P(Positive) - P(Negative)`

## Files

- `app.py` — Streamlit deployment
- `notebook/Untitled23.ipynb` — original notebook
- `requirements.txt` — deployment dependencies

## Example

```text
V, MA, AXP, PYPL
```

## Deployment

Deploy `app.py` through Streamlit Community Cloud with the repository's `requirements.txt`.
