// fomm_engine.h
#pragma once

#include <jni.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <android/bitmap.h>

// JNI string conversion helper, implemented in fomm_engine.cpp,
// used by native_bridge.cpp
std::string jstringToString(JNIEnv* env, jstring jstr);

class FommEngine {
public:
    FommEngine();
    ~FommEngine();

    // Initializes the ONNX sessions. Paths should point to the extracted models on the device.
    bool initialize(const std::string& kpModelPath, const std::string& genModelPath);

    // Core pipeline equivalent to JS runFrame() and computeSourceKeypoints()
    bool processFrame(void* sourcePixels, void* drivingPixels, void* outputPixels, int width, int height);

private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> kpSession;
    std::unique_ptr<Ort::Session> genSession;

    // Tensor dimension constants
    const int64_t BATCH_SIZE = 1;
    const int64_t CHANNELS = 3;
    const int64_t TARGET_SIZE = 256;

    // Reusable allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // Helper functions translating the JS frameToTensor and tensorToImageData logic
    std::vector<float> bitmapToTensor(void* pixels, int width, int height);
    void tensorToBitmap(const float* tensorData, void* outPixels, int width, int height);

    // Keypoint extraction
    struct Keypoints {
        std::vector<float> kp;
        std::vector<float> jac;
        std::vector<int64_t> kp_shape;
        std::vector<int64_t> jac_shape;
    };
    Keypoints extractKeypoints(const std::vector<float>& inputTensor);

    // Helper function to get input/output names from ONNX model
    std::vector<std::string> getInputOutputNames(Ort::Session& session, bool isInput);
};