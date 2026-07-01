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
find "$TEMP_DIR" -type f \( -name "*FOMMDetector.onnx" -o -name "*FOMMDetector.data" \) -exec cp {} "$ASSETS_DIR/" \;
find "$TEMP_DIR" -type f \( -name "*FOMMGenerator.onnx" -o -name "*FOMMGenerator.data" \) -exec cp {} "$ASSETS_DIR/" \;

echo "5. Cleaning up temporary model files..."
rm -rf fomm.zip "$TEMP_DIR"

echo "6. Fetching ONNX Runtime C++ dependencies from Maven Central..."
ONNX_VERSION="1.19.2"
ONNX_AAR_URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ONNX_VERSION}/onnxruntime-android-${ONNX_VERSION}.aar"
ONNX_NATIVE_DIR="app/src/main/cpp/onnxruntime"

mkdir -p "$ONNX_NATIVE_DIR/include"
mkdir -p "$ONNX_NATIVE_DIR/lib"

wget -qO onnxruntime.aar "$ONNX_AAR_URL"
unzip -q onnxruntime.aar -d temp_onnx

# Extract headers gracefully 
if [ -d "temp_onnx/headers" ]; then
    cp -r temp_onnx/headers/* "$ONNX_NATIVE_DIR/include/"
else
    find temp_onnx -type f -name "*.h" -exec cp {} "$ONNX_NATIVE_DIR/include/" \;
fi

# Extract architecture-specific JNI binaries for linking
cp -r temp_onnx/jni/* "$ONNX_NATIVE_DIR/lib/"

rm -rf onnxruntime.aar temp_onnx

echo "Model fetching and native dependency extraction complete."
