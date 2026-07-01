cmake_minimum_required(VERSION 3.22.1)
project("motionforge")

# Find the ONNX Runtime native package exposed via Prefab dependency resolution
find_package(onnxruntime REQUIRED CONFIG)

add_library(
        motionforge
        SHARED
        motionforge-engine.cpp
)

target_link_libraries(
        motionforge
        android
        log
        onnxruntime::onnxruntime
)
