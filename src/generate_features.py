import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

def load_meta(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "meta_ads_campaign_stats.csv"))
    df = df.rename(columns={
        "date_start": "date", "spend": "spend",
        "clicks": "clicks", "impressions": "impressions",
        "conversion": "conversions", "campaign_name": "campaign_name",
        "daily_budget": "daily_budget"
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["channel"] = "Meta"
    df["campaign_type"] = "Social"
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0)
    df["revenue"] = np.nan
    df["campaign_id"] = df["campaign_id"].astype(str)
    return df[["date","channel","campaign_type","campaign_name","campaign_id",
               "spend","clicks","impressions","conversions","daily_budget","revenue"]]

def load_google(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "google_ads_campaign_stats.csv"))
    df = df.rename(columns={
        "segments_date": "date", "metrics_clicks": "clicks",
        "metrics_conversions": "conversions", "metrics_cost_micros": "spend",
        "metrics_impressions": "impressions", "metrics_conversions_value": "revenue",
        "campaign_advertising_channel_type": "campaign_type",
        "campaign_budget_amount": "daily_budget", "campaign_name": "campaign_name"
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["channel"] = "Google"
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0) / 1_000_000
    df["campaign_id"] = df["campaign_id"].astype(str)
    return df[["date","channel","campaign_type","campaign_name","campaign_id",
               "spend","clicks","impressions","conversions","daily_budget","revenue"]]

def load_bing(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "bing_campaign_stats.csv"))
    df = df.rename(columns={
        "TimePeriod": "date", "Spend": "spend", "Clicks": "clicks",
        "Impressions": "impressions", "Conversions": "conversions",
        "Revenue": "revenue", "CampaignType": "campaign_type",
        "DailyBudget": "daily_budget", "CampaignName": "campaign_name",
        "CampaignId": "campaign_id"
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["channel"] = "Bing"
    df["campaign_id"] = df["campaign_id"].astype(str)
    return df[["date","channel","campaign_type","campaign_name","campaign_id",
               "spend","clicks","impressions","conversions","daily_budget","revenue"]]

def validate_campaigns(df):
    issues = []

    # 1. Null date check
    null_dates = df["date"].isna().sum()
    if null_dates > 0:
        issues.append(f"WARNING: {null_dates} rows have invalid/missing dates")
    else:
        issues.append("OK: All date fields are valid")

    # 2. Negative or missing spend
    bad_spend = (df["spend"].isna() | (df["spend"] < 0)).sum()
    if bad_spend > 0:
        issues.append(f"WARNING: {bad_spend} rows have missing or negative spend")
    else:
        issues.append("OK: All spend values are valid")

    # 3. Zero spend campaigns
    zero_spend = df[df["spend"] == 0]["campaign_name"].nunique()
    if zero_spend > 0:
        issues.append(f"WARNING: {zero_spend} campaigns have zero spend — may affect forecast accuracy")
    else:
        issues.append("OK: No zero-spend campaigns detected")

    # 4. Duplicate rows check
    dupes = df.duplicated(subset=["date","campaign_name","channel"]).sum()
    if dupes > 0:
        issues.append(f"WARNING: {dupes} duplicate rows found (same date + campaign + channel)")
    else:
        issues.append("OK: No duplicate rows detected")

    # 5. Missing channels
    for ch in ["Meta", "Google", "Bing"]:
        if ch not in df["channel"].unique():
            issues.append(f"ERROR: {ch} channel data missing entirely")
        else:
            count = len(df[df["channel"] == ch])
            issues.append(f"OK: {ch} channel has {count:,} rows")

    # 6. Spend anomaly detection (values > 3 std deviations)
    mean_spend = df["spend"].mean()
    std_spend  = df["spend"].std()
    anomalies  = df[df["spend"] > mean_spend + 3 * std_spend]
    if len(anomalies) > 0:
        issues.append(f"WARNING: {len(anomalies)} spend anomalies detected (>3σ from mean) — review before forecasting")
    else:
        issues.append("OK: No spend anomalies detected")

    # 7. Date range check
    date_range = (df["date"].max() - df["date"].min()).days
    if date_range < 30:
        issues.append(f"WARNING: Only {date_range} days of data — forecasts may be unreliable")
    else:
        issues.append(f"OK: {date_range} days of historical data available")

    # 8. Cross-channel consistency
    channels_found = df["channel"].unique().tolist()
    if len(channels_found) < 3:
        issues.append(f"WARNING: Only {len(channels_found)} channel(s) found — blended ROAS may be incomplete")
    else:
        issues.append(f"OK: All 3 channels present — {', '.join(channels_found)}")

    # 9. Conversion sanity check
    zero_conv_campaigns = df[df["conversions"] == 0]["campaign_name"].nunique()
    if zero_conv_campaigns > 0:
        issues.append(f"WARNING: {zero_conv_campaigns} campaigns have zero conversions — revenue estimates may be low")

    # 10. Budget consistency
    missing_budget = (df["daily_budget"].isna() | (df["daily_budget"] <= 0)).sum()
    if missing_budget > 0:
        issues.append(f"WARNING: {missing_budget} rows have missing/zero daily budget")
    else:
        issues.append("OK: All daily budget values are present")

    return issues
def build_features(df):
    # Estimate Meta revenue
    known = df[df["revenue"].notna() & (df["conversions"] > 0)]
    avg_rpc = (known["revenue"].sum() / known["conversions"].sum()) if len(known) > 0 else 50.0
    mask = df["channel"] == "Meta"
    df.loc[mask, "revenue"] = df.loc[mask, "conversions"] * avg_rpc
    df["revenue"] = df["revenue"].fillna(0)
    df["spend"] = df["spend"].fillna(0)

    # Derived features
    df["roas"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0)
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0)
    df["conv_rate"] = np.where(df["clicks"] > 0, df["conversions"] / df["clicks"], 0)
    df["cpc"] = np.where(df["clicks"] > 0, df["spend"] / df["clicks"], 0)

    # Time features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Channel encoding
    df["channel_enc"] = df["channel"].map({"Google": 0, "Meta": 1, "Bing": 2}).fillna(0)
    return df

def build_future_rows(df, horizon_days=90):
    """Generate future rows per campaign using rolling averages as inputs"""
    future_rows = []
    latest_date = df["date"].max()
    future_dates = pd.date_range(
        start=latest_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D"
    )

    # Per campaign stats (last 30 days average)
    recent = df[df["date"] >= latest_date - pd.Timedelta(days=30)]
    campaign_stats = recent.groupby(
        ["campaign_name", "campaign_id", "channel", "campaign_type"]
    ).agg(
        avg_spend=("spend", "mean"),
        avg_clicks=("clicks", "mean"),
        avg_impressions=("impressions", "mean"),
        avg_conversions=("conversions", "mean"),
        avg_daily_budget=("daily_budget", "mean"),
        avg_ctr=("ctr", "mean"),
        avg_conv_rate=("conv_rate", "mean"),
        avg_cpc=("cpc", "mean"),
    ).reset_index()

    for _, camp in campaign_stats.iterrows():
        for fdate in future_dates:
            # Seasonality adjustment based on month
            month = fdate.month
            season_factor = {
                1:0.85, 2:0.88, 3:0.92, 4:0.95,
                5:1.00, 6:1.05, 7:1.02, 8:1.00,
                9:1.05, 10:1.10, 11:1.20, 12:1.30
            }.get(month, 1.0)

            future_rows.append({
                "date": fdate,
                "campaign_name": camp["campaign_name"],
                "campaign_id": camp["campaign_id"],
                "channel": camp["channel"],
                "campaign_type": camp["campaign_type"],
                "spend": camp["avg_spend"] * season_factor,
                "clicks": camp["avg_clicks"] * season_factor,
                "impressions": camp["avg_impressions"] * season_factor,
                "conversions": camp["avg_conversions"] * season_factor,
                "daily_budget": camp["avg_daily_budget"],
                "revenue": np.nan,
                "ctr": camp["avg_ctr"],
                "conv_rate": camp["avg_conv_rate"],
                "cpc": camp["avg_cpc"],
                "roas": 0,
                "month": fdate.month,
                "day_of_week": fdate.dayofweek,
                "quarter": fdate.quarter,
                "week_of_year": fdate.isocalendar()[1],
                "channel_enc": {"Google":0,"Meta":1,"Bing":2}.get(camp["channel"], 0),
                "is_future": True
            })

    future_df = pd.DataFrame(future_rows)
    df["is_future"] = False
    return pd.concat([df, future_df], ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="features.parquet")
    args = parser.parse_args()

    meta   = load_meta(args.data_dir)
    google = load_google(args.data_dir)
    bing   = load_bing(args.data_dir)

    df = pd.concat([meta, google, bing], ignore_index=True)

    # Validate
    print("\n── Campaign Validation ──")
    issues = validate_campaigns(df)
    for issue in issues:
        print(f"  {issue}")
    print()

    df = build_features(df)
    df = build_future_rows(df, horizon_days=90)

    df.to_parquet(args.out, index=False)
    historical = df[~df["is_future"]]
    future = df[df["is_future"]]
    print(f"✅ Features saved → {args.out}")
    print(f"   Historical rows : {len(historical):,}")
    print(f"   Future rows     : {len(future):,}")
    print(f"   Total           : {len(df):,}")

if __name__ == "__main__":
    main()