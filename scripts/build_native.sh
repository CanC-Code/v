#!/usr/bin/env bash
set -e

# Source the environment variables
source ./scripts/setup_env.sh

# Define native build paths
# Note: targeting arm64-v8a as the primary 64-bit architecture
TARGET_ABI="arm64-v8a"
BUILD_DIR="app/.cxx/cmake/standalone/$TARGET_ABI"
CMAKE_LISTS_PATH="app/src/main/cpp"

echo "Initiating standalone native CMake build for $TARGET_ABI..."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Invoke CMake with the Android Toolchain
cmake -H"../../../../../$CMAKE_LISTS_PATH" \
      -B"." \
      -DANDROID_ABI="$TARGET_ABI" \
      -DANDROID_NDK="$ANDROID_NDK_HOME" \
      -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
      -DANDROID_PLATFORM=android-26 \
      -DCMAKE_BUILD_TYPE=Release

echo "Compiling native libraries using $(nproc) threads..."
make -j$(nproc)

echo "Native compilation finished successfully. Shared object binaries:"
find . -name "*.so" -exec ls -lh {} \;

# Return to root
cd - > /dev/null
