#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Step 1: Generating features from $DATA_DIR..."
python src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.parquet

echo "Step 2: Running predictions..."
python src/predict.py \
    --features features.parquet \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo "Step 3: Generating AI causal insights..."
python src/generate_insights.py || echo "Insights skipped — using pre-generated files"

echo "Done. Predictions written to $OUTPUT_PATH"