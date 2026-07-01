#!/usr/bin/env bash
set -e

echo "Validating Android Development Environment..."

# 1. Resolve Android SDK
if [ -z "$ANDROID_HOME" ]; then
    if [ -d "$HOME/Android/Sdk" ]; then
        export ANDROID_HOME="$HOME/Android/Sdk"
    else
        echo "Error: ANDROID_HOME is not set and default SDK path was not found."
        exit 1
    fi
fi

# 2. Resolve Android NDK
if [ -z "$ANDROID_NDK_HOME" ]; then
    # Dynamically find the latest NDK installed within the SDK directory
    if [ -d "$ANDROID_HOME/ndk" ]; then
        NDK_DIR=$(find "$ANDROID_HOME/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)
        if [ -n "$NDK_DIR" ]; then
            export ANDROID_NDK_HOME="$NDK_DIR"
        else
            echo "Error: ANDROID_NDK_HOME is not set and no NDK was found in $ANDROID_HOME/ndk."
            exit 1
        fi
    else
        echo "Error: ANDROID_NDK_HOME is not set and the $ANDROID_HOME/ndk directory does not exist."
        exit 1
    fi
fi

# 3. Validate Java Availability
if ! command -v java &> /dev/null; then
    echo "Error: Java is not installed or not present in PATH."
    exit 1
fi

# 4. Validate Gradle (Optional direct wrapper check)
if [ ! -f "./gradlew" ]; then
    echo "Warning: gradlew wrapper not found in the current directory. Ensure you are running this from the project root."
fi

echo "Environment successfully configured:"
echo "ANDROID_HOME:     $ANDROID_HOME"
echo "ANDROID_NDK_HOME: $ANDROID_NDK_HOME"
echo "Java Version:     $(java -version 2>&1 | head -n 1)"
