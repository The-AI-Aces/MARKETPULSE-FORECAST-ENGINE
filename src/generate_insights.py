from groq import Groq
import pandas as pd
import os
import time
import sys
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY :
    print("WARNING: No GROQ_API_KEY found — using pre-generated insight files")
    sys.exit(0)

client = Groq(api_key=GROQ_API_KEY)
os.makedirs("output", exist_ok=True)

predictions_path = "./output/predictions.csv"
if not os.path.exists(predictions_path):
    print("WARNING: predictions.csv not found — run predict.py first")
    sys.exit(0)

df = pd.read_csv(predictions_path)

for window in [30, 60, 90]:
    subset   = df[df["forecast_window_days"] == window]
    ch_level = subset[subset["level"] == "channel"]
    ct_level = subset[subset["level"] == "campaign_type"]

    if ch_level.empty:
        print(f"WARNING: No channel data for {window}d window — skipping")
        continue

    total_spend  = ch_level["total_spend"].sum()
    rev_mid      = ch_level["revenue_mid"].sum()
    rev_low      = ch_level["revenue_low"].sum()
    rev_high     = ch_level["revenue_high"].sum()
    blended_roas = rev_mid / total_spend if total_spend > 0 else 0

    prompt = f"""You are a senior digital marketing analyst.

FORECAST PERIOD: Next {window} days
TOTAL AD SPEND: ${total_spend:,.0f}
REVENUE: ${rev_low:,.0f} (low) to ${rev_mid:,.0f} (mid) to ${rev_high:,.0f} (high)
BLENDED ROAS: {blended_roas:.2f}x

CHANNEL DATA:
{ch_level[['group','total_spend','revenue_low','revenue_mid','revenue_high','roas_mid']].to_string(index=False)}

CAMPAIGN TYPE DATA:
{ct_level[['group','total_spend','revenue_mid','roas_mid']].to_string(index=False)}

Write a causal analysis covering:
## Causal Factors Driving This Forecast
(3 reasons WHY revenue is at this level — use actual numbers)
## Channel Efficiency Analysis
(most/least efficient channel and WHY — use ROAS numbers)
## Seasonality Impact
(how seasonal trends affect this {window}-day period)
## Risk Factors
(2 risks that could cause deviation from forecast)
## Budget Recommendations
(2 actionable recommendations with ROAS impact)
Be specific. Use actual numbers. Focus on causality."""

    print(f"Generating {window}-day insights via Groq...")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )
        text = response.choices[0].message.content
        with open(f"output/insights_{window}d.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Insights saved → output/insights_{window}d.txt")
        time.sleep(3)
    except Exception as e:
        print(f"Error for {window}d: {e}")
        print("Using pre-generated file if available")

print("\nInsights generation complete!")