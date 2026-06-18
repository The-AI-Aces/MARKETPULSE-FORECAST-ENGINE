import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    print("📦 Loading features...")
    df = pd.read_parquet("features.parquet")

    # Train ONLY on historical data
    hist = df[df["is_future"] == False].copy()
    hist = hist[hist["spend"] > 0]

    FEATURES = [
        "spend", "clicks", "impressions", "conversions",
        "daily_budget", "ctr", "conv_rate", "cpc",
        "month", "day_of_week", "quarter", "week_of_year",
        "channel_enc"
    ]
    TARGET = "revenue"

    hist = hist.dropna(subset=FEATURES + [TARGET])
    X = hist[FEATURES]
    y = hist[TARGET]

    print(f"🔢 Training on {len(X):,} historical rows...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"✅ MAE : {mae:.2f}")
    print(f"✅ R²  : {r2:.4f}")

    # Feature importance
    print("\n── Top 5 Features ──")
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    for feat, score in importances.nlargest(5).items():
        print(f"   {feat:<20} {score:.4f}")

    os.makedirs("pickle", exist_ok=True)
    joblib.dump(model, "pickle/model.pkl")
    joblib.dump(FEATURES, "pickle/features.pkl")
    print("\n💾 Model saved → pickle/model.pkl")

if __name__ == "__main__":
    main()