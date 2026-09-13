# 🏦 Institutional Financial NLP Sentiment Benchmark

This repository contains the original notebook and a Streamlit deployment version of the same financial-news sentiment pipeline.

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
