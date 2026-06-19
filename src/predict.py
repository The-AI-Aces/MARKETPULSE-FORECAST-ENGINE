import pandas as pd
import numpy as np
import joblib
import argparse
import os
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.parquet")
    parser.add_argument("--model",    default="./pickle/model.pkl")
    parser.add_argument("--output",   default="./output/predictions.csv")
    args = parser.parse_args()

    print("Loading model...")
    model    = joblib.load(args.model)
    FEATURES = joblib.load("./pickle/features.pkl")

    print("Loading future rows...")
    df = pd.read_parquet(args.features)
    future = df[df["is_future"] == True].copy()

    if future.empty:
        print("WARNING: No future rows found — check data/ folder")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        pd.DataFrame().to_csv(args.output, index=False)
        return

    future["daily_budget"] = future["daily_budget"].fillna(0)
    future["conversions"]  = future["conversions"].fillna(0)
    future["clicks"]       = future["clicks"].fillna(0)
    future["impressions"]  = future["impressions"].fillna(0)
    future = future[future["spend"] > 0].dropna(subset=FEATURES)

    if future.empty:
        print("WARNING: No valid future rows after filtering")
        pd.DataFrame().to_csv(args.output, index=False)
        return

    print(f"Forecasting {len(future):,} future data points...")
    future["predicted_revenue"] = model.predict(future[FEATURES])
    future["revenue_low"]       = future["predicted_revenue"] * 0.85
    future["revenue_mid"]       = future["predicted_revenue"]
    future["revenue_high"]      = future["predicted_revenue"] * 1.15
    future["roas_low"]  = np.where(future["spend"] > 0, future["revenue_low"]  / future["spend"], 0)
    future["roas_mid"]  = np.where(future["spend"] > 0, future["revenue_mid"]  / future["spend"], 0)
    future["roas_high"] = np.where(future["spend"] > 0, future["revenue_high"] / future["spend"], 0)

    results = []
    for window in [30, 60, 90]:
        cutoff = future["date"].min() + pd.Timedelta(days=window)
        subset = future[future["date"] <= cutoff]
        if subset.empty:
            continue

        # --- Channel level ---
        for ch in subset["channel"].unique():
            c = subset[subset["channel"] == ch]
            results.append({
                "forecast_window_days": window, "level": "channel",
                "group": ch, "campaign_type": "ALL", "campaign_name": "ALL",
                "total_spend":  round(c["spend"].sum(), 2),
                "revenue_low":  round(c["revenue_low"].sum(), 2),
                "revenue_mid":  round(c["revenue_mid"].sum(), 2),
                "revenue_high": round(c["revenue_high"].sum(), 2),
                "roas_low":     round(c["roas_low"].mean(), 4),
                "roas_mid":     round(c["roas_mid"].mean(), 4),
                "roas_high":    round(c["roas_high"].mean(), 4),
            })

        # --- Campaign type level ---
        for ct in subset["campaign_type"].unique():
            c = subset[subset["campaign_type"] == ct]
            results.append({
                "forecast_window_days": window, "level": "campaign_type",
                "group": ct, "campaign_type": ct, "campaign_name": "ALL",
                "total_spend":  round(c["spend"].sum(), 2),
                "revenue_low":  round(c["revenue_low"].sum(), 2),
                "revenue_mid":  round(c["revenue_mid"].sum(), 2),
                "revenue_high": round(c["revenue_high"].sum(), 2),
                "roas_low":     round(c["roas_low"].mean(), 4),
                "roas_mid":     round(c["roas_mid"].mean(), 4),
                "roas_high":    round(c["roas_high"].mean(), 4),
            })

        # --- Campaign level ---
        for cn in subset["campaign_name"].unique():
            c = subset[subset["campaign_name"] == cn]
            ct = c["campaign_type"].iloc[0] if not c.empty else "UNKNOWN"
            ch = c["channel"].iloc[0] if not c.empty else "UNKNOWN"
            results.append({
                "forecast_window_days": window, "level": "campaign",
                "group": cn, "campaign_type": ct, "campaign_name": cn,
                "total_spend":  round(c["spend"].sum(), 2),
                "revenue_low":  round(c["revenue_low"].sum(), 2),
                "revenue_mid":  round(c["revenue_mid"].sum(), 2),
                "revenue_high": round(c["revenue_high"].sum(), 2),
                "roas_low":     round(c["roas_low"].mean(), 4),
                "roas_mid":     round(c["roas_mid"].mean(), 4),
                "roas_high":    round(c["roas_high"].mean(), 4),
            })

    # --- Write output ---
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()