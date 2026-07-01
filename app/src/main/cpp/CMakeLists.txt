cmake_minimum_required(VERSION 3.22.1)
project("motionforge_native")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the ONNX Runtime package provided by the AAR via Prefab
find_package(onnxruntime REQUIRED CONFIG)

# Find Android-specific libraries for logging and Bitmap manipulation
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

# Define our native engine library
add_library(motionforge_engine SHARED
            native_bridge.cpp
            fomm_engine.cpp)

# Link dependencies
target_link_libraries(motionforge_engine
                      onnxruntime::onnxruntime
                      ${log-lib}
                      ${jnigraphics-lib})
