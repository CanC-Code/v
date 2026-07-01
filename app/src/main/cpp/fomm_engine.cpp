// fomm_engine.cpp
#include "fomm_engine.h"
#include <android/log.h>
#include <stdexcept>
#include <algorithm>
#include <vector>

#define LOG_TAG "FommEngine"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Constructor
FommEngine::FommEngine()
    : env(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FommEngine")),
      allocator(Ort::AllocatorWithDefaultOptions()) {}

// Destructor
FommEngine::~FommEngine() {
    if (kpSession) kpSession->release();
    if (genSession) genSession->release();
}

// Helper function to get input/output names from ONNX model
std::vector<std::string> FommEngine::getInputOutputNames(Ort::Session& session, bool isInput) {
    std::vector<std::string> names;
    size_t numNodes = isInput ? session.GetInputCount() : session.GetOutputCount();
    for (size_t i = 0; i < numNodes; ++i) {
        // Directly assign the result to Ort::AllocatedStringPtr
        Ort::AllocatedStringPtr namePtr = isInput ?
            session.GetInputNameAllocated(i, allocator) :
            session.GetOutputNameAllocated(i, allocator);
        names.emplace_back(namePtr.get());
    }
    return names;
}

// Helper function to convert std::vector<std::string> to const char* const*
std::vector<const char*> stringVecToCharPtrVec(const std::vector<std::string>& strings) {
    std::vector<const char*> charPtrs;
    for (const auto& str : strings) {
        charPtrs.push_back(str.c_str());
    }
    return charPtrs;
}

// Initialize ONNX sessions
bool FommEngine::initialize(const std::string& kpModelPath, const std::string& genModelPath) {
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);

        kpSession = std::make_unique<Ort::Session>(*env, kpModelPath.c_str(), sessionOptions);
        genSession = std::make_unique<Ort::Session>(*env, genModelPath.c_str(), sessionOptions);

        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to initialize ONNX sessions: %s", e.what());
        return false;
    }
}

// Convert bitmap pixels to tensor
std::vector<float> FommEngine::bitmapToTensor(void* pixels, int width, int height) {
    // Placeholder: Implement your actual conversion logic here
    std::vector<float> tensor(TARGET_SIZE * TARGET_SIZE * CHANNELS, 0.0f);
    return tensor;
}

// Convert tensor to bitmap pixels
void FommEngine::tensorToBitmap(const float* tensorData, void* outPixels, int width, int height) {
    // Placeholder: Implement your actual conversion logic here
}

// Extract keypoints from input tensor
FommEngine::Keypoints FommEngine::extractKeypoints(const std::vector<float>& inputTensor) {
    Keypoints keypoints;
    try {
        // Dynamically discover input/output names for kpSession
        auto kpInputNames = getInputOutputNames(*kpSession, true);
        auto kpOutputNames = getInputOutputNames(*kpSession, false);

        // Convert to const char* const*
        auto kpInputNamesPtr = stringVecToCharPtrVec(kpInputNames);
        auto kpOutputNamesPtr = stringVecToCharPtrVec(kpOutputNames);

        // Create input tensor
        std::vector<int64_t> inputShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensorValue = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(inputTensor.data()), inputTensor.size(), inputShape.data(), inputShape.size());

        // Prepare output tensors
        std::vector<float> kpOutput(TARGET_SIZE * TARGET_SIZE * 2); // Example size, adjust as needed
        std::vector<float> jacOutput(TARGET_SIZE * TARGET_SIZE * 2); // Example size, adjust as needed
        std::vector<int64_t> kpShape = {BATCH_SIZE, TARGET_SIZE, TARGET_SIZE, 2}; // Example shape
        std::vector<int64_t> jacShape = {BATCH_SIZE, TARGET_SIZE, TARGET_SIZE, 2}; // Example shape

        Ort::Value kpOutputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, kpOutput.data(), kpOutput.size(), kpShape.data(), kpShape.size());
        Ort::Value jacOutputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, jacOutput.data(), jacOutput.size(), jacShape.data(), jacShape.size());

        // Run kpSession
        Ort::Value outputTensors[] = {kpOutputTensor, jacOutputTensor};
        kpSession->Run(
            Ort::RunOptions{nullptr},
            kpInputNamesPtr.data(), &inputTensorValue, 1,
            kpOutputNamesPtr.data(), outputTensors, 2
        );

        keypoints.kp = kpOutput;
        keypoints.jac = jacOutput;
        keypoints.kp_shape = kpShape;
        keypoints.jac_shape = jacShape;
    } catch (const std::exception& e) {
        LOGE("Exception in extractKeypoints: %s", e.what());
    }
    return keypoints;
}

// Core pipeline: process a frame
bool FommEngine::processFrame(void* sourcePixels, void* drivingPixels, void* outputPixels, int width, int height) {
    try {
        // Convert bitmaps to tensors
        std::vector<float> sourceTensor = bitmapToTensor(sourcePixels, width, height);
        std::vector<float> drivingTensor = bitmapToTensor(drivingPixels, width, height);

        // Extract keypoints for source and driving frames
        Keypoints sourceKeypoints = extractKeypoints(sourceTensor);
        Keypoints drivingKeypoints = extractKeypoints(drivingTensor);

        // Dynamically discover input/output names for genSession
        auto genInputNames = getInputOutputNames(*genSession, true);
        auto genOutputNames = getInputOutputNames(*genSession, false);

        // Convert to const char* const*
        auto genInputNamesPtr = stringVecToCharPtrVec(genInputNames);
        auto genOutputNamesPtr = stringVecToCharPtrVec(genOutputNames);

        // Prepare input tensors for genSession
        std::vector<int64_t> inputShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensorValue = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(sourceTensor.data()), sourceTensor.size(), inputShape.data(), inputShape.size());

        // Prepare output tensor for genSession
        std::vector<float> outputTensor(TARGET_SIZE * TARGET_SIZE * CHANNELS);
        std::vector<int64_t> outputShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
        Ort::Value outputTensorValue = Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensor.data(), outputTensor.size(), outputShape.data(), outputShape.size());

        // Run genSession
        genSession->Run(
            Ort::RunOptions{nullptr},
            genInputNamesPtr.data(), &inputTensorValue, 1,
            genOutputNamesPtr.data(), &outputTensorValue, 1
        );

        // Convert output tensor to bitmap
        tensorToBitmap(outputTensor.data(), outputPixels, width, height);
        return true;
    } catch (const std::exception& e) {
        LOGE("Exception in processFrame: %s", e.what());
        return false;
    }
}