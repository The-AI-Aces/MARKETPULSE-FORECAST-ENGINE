# 📈 MarketPulse Forecast Engine

**Probabilistic Revenue & ROAS Forecasting for E-commerce Marketing**

[![Hackathon](https://img.shields.io/badge/AIgnition-3.0-0E9488)](https://github.com/The-AI-Aces/MARKETPULSE-FORECAST-ENGINE)
[![Team](https://img.shields.io/badge/Team-The%20AI%20Aces-B45309)](https://github.com/The-AI-Aces)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](#)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)](#)
[![Model](https://img.shields.io/badge/Model-RandomForestRegressor-green)](#)
[![LLM](https://img.shields.io/badge/LLM-Groq%20Llama--3.3--70B-orange)](#)

> **Team:** The AI Aces &nbsp;·&nbsp; **Hackathon:** AIgnition 3.0 by NetElixir &nbsp;·&nbsp; **Deadline:** July 19, 2026

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [What MarketPulse Does](#what-marketpulse-does)
3. [Architecture](#architecture)
4. [Repository Structure](#repository-structure)
5. [Quick Start — Automated Grading](#quick-start--automated-grading)
6. [Interactive Demo](#interactive-demo)
7. [Forecasting Methodology](#forecasting-methodology)
8. [Data Validation](#data-validation)
9. [AI Integration Strategy](#ai-integration-strategy)
10. [Assumptions & Limitations](#assumptions--limitations)
11. [Evaluation Criteria Mapping](#evaluation-criteria-mapping)
12. [Team](#team)

---

## Problem Statement

E-commerce agencies run budgets simultaneously across Google Ads, Meta Ads, and Microsoft (Bing) Ads. Forecasting how that combined spend translates into future revenue is still mostly done in spreadsheets — manually, per channel, with no unified view of uncertainty and no explanation of *why* a number is expected to move.

**MarketPulse Forecast Engine** is built for that exact gap. Given historical multi-channel campaign performance data, it produces **probabilistic** (Low / Mid / High) revenue and ROAS forecasts for the next 30, 60, or 90 days — broken down by channel, campaign type, and individual campaign — and generates an LLM-powered causal explanation of *why*, not just a number on a chart.

---

## What MarketPulse Does

| Requirement (AIgnition 3.0 brief) | How MarketPulse delivers it |
|---|---|
| Aggregate e-commerce revenue forecast (Google + Meta + Bing) | Blended revenue forecast across all three channels, summed from per-campaign predictions |
| Blended ROAS forecast | Forecasted revenue ÷ forecasted spend across the active channel set |
| Channel / campaign-type / campaign-level breakdown | Three dedicated views: channel roll-up, campaign-type roll-up, sortable per-campaign table |
| 30 / 60 / 90-day forecasting windows | Selectable in the sidebar — all KPIs, charts, and tables recompute instantly |
| Probabilistic ranges, not single point estimates | Every forecast delivered as **Low / Mid / High** with visual range display |
| Budget simulation | Per-channel multiplier sliders (0.5×–3.0×) instantly re-forecast revenue and ROAS |
| Seasonality awareness | Monthly seasonality multipliers derived from historical patterns, applied at projection time |
| AI-assisted causal inference | Groq Llama-3.3-70B generates plain-English causal breakdowns: drivers, channel efficiency, risks, budget recommendations |

---

## Architecture

```
┌──────────────────────┐     ┌──────────────────────────┐     ┌───────────────────────┐
│   Raw Channel CSVs    │     │    Feature Pipeline       │     │   Forecast Engine      │
│                       │     │                           │     │                        │
│  google_ads_campaign  │────▶│  generate_features.py     │────▶│  RandomForestRegressor │
│  meta_ads_campaign    │     │  · normalize schemas      │     │  · 200 trees           │
│  bing_campaign_stats  │     │  · validate data          │     │  · Low/Mid/High bands  │
└──────────────────────┘     │  · engineer features      │     │  · 30/60/90d windows   │
                              │  · project future rows    │     └──────────┬────────────┘
                              └──────────────────────────┘                │
                                                                           ▼
                              ┌──────────────────────────┐     ┌───────────────────────┐
                              │   Streamlit UI (app.py)   │◀────│  Aggregated Forecasts  │
                              │   · Channel overview      │     │  · Channel level       │
                              │   · Campaign drill-down   │     │  · Campaign type level │
                              │   · Budget simulator      │     │  · Campaign level      │
                              │   · Trend analysis        │     └──────────┬────────────┘
                              │   · AI Insights tab       │                │
                              └──────────────────────────┘                ▼
                                                                ┌───────────────────────┐
                                                                │  Groq Llama-3.3-70B    │
                                                                │  Causal insight        │
                                                                │  generation            │
                                                                └───────────────────────┘
```

**Frontend:** Streamlit — single-page app, Plotly charts, custom light theme
**Backend / ML:** Python · pandas · NumPy · scikit-learn · pyarrow
**LLM Layer:** Groq API (Llama-3.3-70B) with pre-generated offline fallback files
**Storage:** Flat-file pipeline — CSV in → Parquet features → pickled model → CSV/TXT out. Zero infrastructure required.

---

## Repository Structure

```
MARKETPULSE-FORECAST-ENGINE/
│
├── run.sh                          # ← Single entry point for automated grading
├── requirements.txt                # ← Pinned Python dependencies
├── app.py                          # ← Streamlit interactive demo (5 tabs)
│
├── data/                           # ← Input CSVs (replaced by grader at test time)
│   ├── google_ads_campaign_stats.csv
│   ├── meta_ads_campaign_stats.csv
│   └── bing_campaign_stats.csv
│
├── pickle/
│   ├── model.pkl                   # ← Trained RandomForestRegressor (committed)
│   └── features.pkl                # ← Ordered feature list for train/predict parity
│
├── src/
│   ├── generate_features.py        # ← Load, clean, validate, engineer, project
│   ├── predict.py                  # ← Score future rows, aggregate, write CSV
│   ├── generate_insights.py        # ← Groq LLM causal summary generation
│   └── train_model.py              # ← Offline training (NOT run at grading time)
│
├── output/
│   ├── predictions.csv             # ← Forecast output (written fresh every run)
│   ├── insights_30d.txt            # ← Pre-generated AI insights (offline-safe)
│   ├── insights_60d.txt
│   └── insights_90d.txt
│
├── .streamlit/
│   └── config.toml                 # ← Streamlit theme config
│
└── README.md
```

---

## Quick Start — Automated Grading

This follows the **exact contract** in the Hackathon Submission Guide: one command, three positional arguments, sensible defaults.

```bash
# Make executable (if needed)
chmod +x run.sh

# Run with defaults — what the grader runs
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

`run.sh` runs both required pipeline stages in a single invocation:

**Stage 1 — Feature generation**
```bash
python src/generate_features.py --data-dir ./data --out features.parquet
```
Reads all CSVs in `DATA_DIR`, normalizes schemas, validates data, engineers features, and projects forward rows for each campaign.

**Stage 2 — Prediction**
```bash
python src/predict.py --features features.parquet --model ./pickle/model.pkl --output ./output/predictions.csv
```
Loads the pickled model, scores all future rows, aggregates to channel / campaign-type / campaign level, and writes the forecast CSV.

| Argument | Description | Default |
|---|---|---|
| `DATA_DIR` | Folder with the three channel CSVs | `./data` |
| `MODEL_PATH` | Path to pickled model | `./pickle/model.pkl` |
| `OUTPUT_PATH` | Where predictions are written | `./output/predictions.csv` |

**The grading pipeline:**
```
git clone → pip install -r requirements.txt → drop test data into data/ → ./run.sh → read output/predictions.csv
```

---

## Interactive Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Launches the full **MarketPulse** console with 5 tabs:

| Tab | What it shows |
|---|---|
| Channel Overview | Blended revenue and ROAS forecast by channel for selected window |
| Campaign Breakdown | Campaign-type and campaign-level forecast tables, sortable by revenue |
| Budget Simulator | Per-channel spend multiplier sliders with live revenue/ROAS re-forecast |
| Trend Analysis | Historical performance trends overlaid with forecast range |
| AI Insights | LLM-generated causal analysis: drivers, efficiency, risks, recommendations |

**Python version:** 3.11+

---

## Forecasting Methodology

### Feature Engineering
Each channel's raw CSV is normalized into a common schema:
`date · channel · campaign_type · campaign_name · spend · clicks · impressions · conversions · daily_budget · revenue`

Handling channel quirks: Google Ads reports cost in micros (÷1,000,000), Bing reports revenue directly, Meta does not include a revenue column (see *Assumptions*). Derived features include ROAS, CTR, conversion rate, CPC, and calendar signals (month, day-of-week, quarter, week-of-year).

### Model
A `RandomForestRegressor` (200 trees, max depth 12, min 2 samples per leaf) trained on **~25,500 campaign-day records** spanning **136 campaigns** across Google, Meta, and Bing from **January 2024 to June 2026** (~2.5 years). Random Forest was chosen because campaign-level marketing data is noisy and non-linear, channel behavior interacts with seasonality in ways a tree ensemble captures naturally, and training is fast enough to fully retrain within a hackathon timeline. Full MMM and custom attribution were explicitly out of scope per the challenge brief.

### Future Projection & Seasonality
For each active campaign, the last 30 days of activity are averaged into a baseline daily projection, then scaled by a month-level seasonality multiplier (November/December up for holiday demand, January/February down) before being scored by the trained model.

### Uncertainty Bands
Low / Mid / High = point prediction × 0.85 / 1.0 / 1.15. Intentionally simple and transparent — chosen for explainability within the hackathon timeline. The natural upgrade path is quantile regression or bootstrapped residuals for data-driven per-campaign intervals.

### Budget Simulation
The same trained model re-scores "what-if" feature rows where spend, clicks, impressions, and conversions are scaled by a user-chosen multiplier — no retraining required, instant results.

---

## Data Validation

Before any forecast is generated, the data passes a 10-point validation check shown in an expandable **Campaign Validation Report** in the UI:

1. Null / invalid date detection
2. Negative or missing spend detection
3. Zero-spend campaign flagging
4. Duplicate row detection (same date + campaign + channel)
5. Channel completeness check (Google / Meta / Bing all present)
6. Spend anomaly detection (>3σ from channel mean)
7. Minimum historical window check (warns if < 30 days)
8. Cross-channel consistency check (for blended ROAS reliability)
9. Zero-conversion campaign flagging
10. Budget-field completeness check

---

## AI Integration Strategy

After the forecast is produced, the structured output (total spend, revenue range, blended ROAS, channel breakdown, campaign-type breakdown) is passed to **Groq's Llama-3.3-70B** with a prompt requesting five specific sections — all required to cite the actual numbers:

- Causal Factors Driving This Forecast
- Channel Efficiency Analysis
- Seasonality Impact
- Risk Factors
- Budget Recommendations

**Offline-safe design:** `generate_insights.py` is run once during development and its output is committed to `output/insights_{30,60,90}d.txt`. The Streamlit app reads these pre-generated files, so the AI Insights tab works identically with or without a live API key or network access at grading time. If `generate_insights.py` fails during `run.sh`, the pipeline catches the error gracefully and continues — the predictions CSV is always written regardless.

**API key security:** The Groq API key is stored only in a local `.env` file (gitignored) and read at runtime via `os.environ.get("GROQ_API_KEY", "")`. No secrets are committed to the repository.

---

## Assumptions & Limitations

- **Meta revenue is imputed.** Meta's export has no revenue column. Revenue is estimated as `conversions × average revenue-per-conversion` (ratio learned from Google + Bing). This is the primary source of Meta forecast uncertainty and the first thing to replace with real Conversions API data in production.
- **Uncertainty bands are fixed calibration, not learned.** The ±15% envelope is transparent and simple by design; see *Methodology* for the upgrade path.
- **Existing attribution is treated as ground truth** per the brief's explicit constraint — no custom attribution model or MMM is built.
- **Forecasts are aggregate-period, not daily** — matching the 30/60/90-day planning windows the brief specifies.
- **No retraining at grading time.** Model is trained once and committed as a pickle artifact; the graded run only generates features and predicts.

---

