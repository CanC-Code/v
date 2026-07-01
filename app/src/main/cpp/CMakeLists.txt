cmake_minimum_required(VERSION 3.22.1)
project("motionforge_native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -frtti")

# Resolve exact paths using the Gradle-provided extraction directory
set(ONNX_INCLUDE_DIR "${ONNX_NATIVE_DIR}/headers")
set(ONNX_LIB_DIR "${ONNX_NATIVE_DIR}/jni/${ANDROID_ABI}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNX_LIB_DIR}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNX_INCLUDE_DIR}"
)

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

add_library(motionforge_engine SHARED
            native_bridge.cpp
            fomm_engine.cpp)

target_link_libraries(motionforge_engine
                      onnxruntime
                      ${log-lib}
                      ${jnigraphics-lib})

set_target_properties(motionforge_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

set_target_properties(motionforge_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/jniLibs/${ANDROID_ABI}
)
