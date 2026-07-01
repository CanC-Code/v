#!/usr/bin/env bash
set -e

ASSETS_DIR="app/src/main/assets"
TEMP_DIR="temp_models"
MODEL_URL="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/fomm/releases/v0.49.1/fomm-onnx-float.zip"

echo "1. Preparing assets directory..."
mkdir -p "$ASSETS_DIR"
mkdir -p "$TEMP_DIR"

echo "2. Downloading FOMM ONNX bundle..."
wget -qO fomm.zip "$MODEL_URL"

echo "3. Extracting archive..."
unzip -q fomm.zip -d "$TEMP_DIR"

echo "4. Migrating required models and weights to Android assets..."
# Copy files EXACTLY as named to preserve external data metadata links
find "$TEMP_DIR" -type f \( -name "*FOMMDetector.onnx" -o -name "*FOMMDetector.data" \) -exec cp {} "$ASSETS_DIR/" \;
find "$TEMP_DIR" -type f \( -name "*FOMMGenerator.onnx" -o -name "*FOMMGenerator.data" \) -exec cp {} "$ASSETS_DIR/" \;

echo "5. Cleaning up temporary files..."
rm -rf fomm.zip "$TEMP_DIR"

echo "Model fetching complete. Validating assets directory:"
ls -lh "$ASSETS_DIR"
