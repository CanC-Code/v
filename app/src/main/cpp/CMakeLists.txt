cmake_minimum_required(VERSION 3.22.1)
project("motionforge_native")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to the manually extracted ONNX Runtime directory
set(ONNX_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)

# Include ONNX headers
include_directories(${ONNX_RUNTIME_DIR}/include)

# Add ONNX Runtime as an IMPORTED shared library
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNX_RUNTIME_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so
)

# Find Android-specific libraries for logging and Bitmap manipulation
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

# Define our native engine library
add_library(motionforge_engine SHARED
            native_bridge.cpp
            fomm_engine.cpp)

# Link the imported libraries against the engine
target_link_libraries(motionforge_engine
                      onnxruntime
                      ${log-lib}
                      ${jnigraphics-lib})
