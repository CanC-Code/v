// fomm_engine.cpp
#include "fomm_engine.h"
#include <android/log.h>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <utility>

#define LOG_TAG "FommEngine"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global FommEngine instance
FommEngine* gFommEngine = nullptr;

// Helper function: Convert jstring to std::string safely
std::string jstringToString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    if (!chars) return "";
    std::string str(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return str;
}

// Constructor
FommEngine::FommEngine()
    : env(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FommEngine")),
      allocator(Ort::AllocatorWithDefaultOptions()) {
    LOGD("FommEngine constructed");
}

// Destructor
FommEngine::~FommEngine() {
    LOGD("FommEngine destructor");
    if (kpSession) kpSession->release();
    if (genSession) genSession->release();
}

// Helper function to get input/output names from ONNX model
std::vector<std::string> FommEngine::getInputOutputNames(Ort::Session& session, bool isInput) {
    std::vector<std::string> names;
    size_t numNodes = isInput ? session.GetInputCount() : session.GetOutputCount();
    for (size_t i = 0; i < numNodes; ++i) {
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

        LOGD("ONNX sessions initialized successfully");
        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to initialize ONNX sessions: %s", e.what());
        return false;
    }
}

// Convert bitmap pixels to tensor (Alignment safe memcpy)
std::vector<float> FommEngine::bitmapToTensor(void* pixels, int width, int height) {
    std::vector<float> tensor(BATCH_SIZE * CHANNELS * TARGET_SIZE * TARGET_SIZE, 0.0f);
    if (!pixels) return tensor;

    uint8_t* bytePixels = static_cast<uint8_t*>(pixels);
    for (int y = 0; y < std::min(height, static_cast<int>(TARGET_SIZE)); ++y) {
        for (int x = 0; x < std::min(width, static_cast<int>(TARGET_SIZE)); ++x) {
            int srcIdx = y * width + x;
            int dstIdx = (y * TARGET_SIZE + x) * CHANNELS;

            uint32_t pixel = 0;
            std::memcpy(&pixel, bytePixels + srcIdx * 4, 4);

            float r = static_cast<float>((pixel >> 16) & 0xFF) / 255.0f;
            float g = static_cast<float>((pixel >> 8) & 0xFF) / 255.0f;
            float b = static_cast<float>(pixel & 0xFF) / 255.0f;

            tensor[dstIdx] = r;
            tensor[dstIdx + 1] = g;
            tensor[dstIdx + 2] = b;
        }
    }
    return tensor;
}

// Convert tensor to bitmap pixels (Alignment safe memcpy)
void FommEngine::tensorToBitmap(const float* tensorData, void* outPixels, int width, int height) {
    if (!outPixels || !tensorData) return;

    uint8_t* bytePixels = static_cast<uint8_t*>(outPixels);
    for (int y = 0; y < std::min(height, static_cast<int>(TARGET_SIZE)); ++y) {
        for (int x = 0; x < std::min(width, static_cast<int>(TARGET_SIZE)); ++x) {
            int srcIdx = (y * TARGET_SIZE + x) * CHANNELS;
            int dstIdx = y * width + x;

            float r = std::max(0.0f, std::min(1.0f, tensorData[srcIdx]));
            float g = std::max(0.0f, std::min(1.0f, tensorData[srcIdx + 1]));
            float b = std::max(0.0f, std::min(1.0f, tensorData[srcIdx + 2]));
            
            uint8_t r8 = static_cast<uint8_t>(r * 255.0f);
            uint8_t g8 = static_cast<uint8_t>(g * 255.0f);
            uint8_t b8 = static_cast<uint8_t>(b * 255.0f);
            
            uint32_t pixel = (0xFF << 24) | (r8 << 16) | (g8 << 8) | b8;
            std::memcpy(bytePixels + dstIdx * 4, &pixel, 4);
        }
    }
}

// Extract keypoints using dynamic output retrieval to prevent array bounds segfaults
FommEngine::Keypoints FommEngine::extractKeypoints(const std::vector<float>& inputTensor) {
    Keypoints keypoints;
    if (!kpSession) return keypoints;

    try {
        auto kpInputNames = getInputOutputNames(*kpSession, true);
        auto kpOutputNames = getInputOutputNames(*kpSession, false);
        if (kpInputNames.empty() || kpOutputNames.empty()) return keypoints;

        auto kpInputNamesPtr = stringVecToCharPtrVec(kpInputNames);
        auto kpOutputNamesPtr = stringVecToCharPtrVec(kpOutputNames);

        std::vector<int64_t> inputShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value inputTensorValue = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(inputTensor.data()), inputTensor.size(), inputShape.data(), inputShape.size());

        // Properly initialized Ort::RunOptions() avoids the RunOptions{nullptr} crash
        auto outputTensors = kpSession->Run(
            Ort::RunOptions(), 
            kpInputNamesPtr.data(), &inputTensorValue, 1,
            kpOutputNamesPtr.data(), kpOutputNamesPtr.size() 
        );

        if (!outputTensors.empty()) {
            float* kpData = outputTensors[0].GetTensorMutableData<float>();
            size_t kpSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            keypoints.kp.assign(kpData, kpData + kpSize);
            
            if (outputTensors.size() > 1) {
                float* jacData = outputTensors[1].GetTensorMutableData<float>();
                size_t jacSize = outputTensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
                keypoints.jac.assign(jacData, jacData + jacSize);
            }
        }
    } catch (const std::exception& e) {
        LOGE("Exception in extractKeypoints: %s", e.what());
    }
    return keypoints;
}

// Process frame dynamically matching inputs to the generator requirements
bool FommEngine::processFrame(void* sourcePixels, void* drivingPixels, void* outputPixels, int width, int height) {
    if (!kpSession || !genSession) return false;

    try {
        std::vector<float> sourceTensor = bitmapToTensor(sourcePixels, width, height);
        std::vector<float> drivingTensor = bitmapToTensor(drivingPixels, width, height);

        Keypoints sourceKeypoints = extractKeypoints(sourceTensor);
        Keypoints drivingKeypoints = extractKeypoints(drivingTensor);

        auto genInputNames = getInputOutputNames(*genSession, true);
        auto genOutputNames = getInputOutputNames(*genSession, false);
        if (genInputNames.empty() || genOutputNames.empty()) return false;

        auto genInputNamesPtr = stringVecToCharPtrVec(genInputNames);
        auto genOutputNamesPtr = stringVecToCharPtrVec(genOutputNames);

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> inputTensors;

        // Input 0: Source Image
        std::vector<int64_t> imgShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, sourceTensor.data(), sourceTensor.size(), imgShape.data(), imgShape.size()));

        // Bind subsequent expected inputs (kp_driving, kp_source, jacobians) based on graph requirements
        if (genInputNames.size() > 1 && !drivingKeypoints.kp.empty() && !sourceKeypoints.kp.empty()) {
            int64_t num_kp = drivingKeypoints.kp.size() / 2;
            std::vector<int64_t> kpShape = {BATCH_SIZE, num_kp, 2};
            
            inputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, drivingKeypoints.kp.data(), drivingKeypoints.kp.size(), kpShape.data(), kpShape.size()));
                
            if (genInputNames.size() > 2) {
                inputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, sourceKeypoints.kp.data(), sourceKeypoints.kp.size(), kpShape.data(), kpShape.size()));
            }
            if (genInputNames.size() > 3 && !drivingKeypoints.jac.empty()) {
                 std::vector<int64_t> jacShape = {BATCH_SIZE, num_kp, 2, 2};
                 inputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, drivingKeypoints.jac.data(), drivingKeypoints.jac.size(), jacShape.data(), jacShape.size()));
            }
            if (genInputNames.size() > 4 && !sourceKeypoints.jac.empty()) {
                 std::vector<int64_t> jacShape = {BATCH_SIZE, num_kp, 2, 2};
                 inputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, sourceKeypoints.jac.data(), sourceKeypoints.jac.size(), jacShape.data(), jacShape.size()));
            }
        }

        size_t inputCount = std::min(inputTensors.size(), genInputNamesPtr.size());
        
        auto outputTensors = genSession->Run(
            Ort::RunOptions(),
            genInputNamesPtr.data(), inputTensors.data(), inputCount,
            genOutputNamesPtr.data(), genOutputNamesPtr.size()
        );

        if (!outputTensors.empty()) {
            float* outData = outputTensors[0].GetTensorMutableData<float>();
            tensorToBitmap(outData, outputPixels, width, height);
            LOGD("Frame processed successfully");
            return true;
        }

        return false;
    } catch (const std::exception& e) {
        LOGE("Exception in processFrame: %s", e.what());
        return false;
    }
}
