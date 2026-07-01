cmake_minimum_required(VERSION 3.22.1)
project("motionforge")

# 1. Resolve exact paths using the Gradle-provided extraction directory
set(ONNX_INCLUDE_DIR "${ONNX_NATIVE_DIR}/headers")
set(ONNX_LIB_DIR "${ONNX_NATIVE_DIR}/jni/${ANDROID_ABI}")

add_library(
        motionforge
        SHARED
        motionforge-engine.cpp
)

# 2. Expose the ONNX headers to the C++ compiler
target_include_directories(motionforge PRIVATE ${ONNX_INCLUDE_DIR})

# 3. Import the prebuilt shared library object (.so) for the current target architecture
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNX_LIB_DIR}/libonnxruntime.so")

# 4. Link the native pipeline
target_link_libraries(
        motionforge
        android
        log
        onnxruntime
)
