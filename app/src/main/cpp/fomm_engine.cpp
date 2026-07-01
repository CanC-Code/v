// fomm_engine.cpp
#include "fomm_engine.h"
#include <android/log.h>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <utility>
#include <jni.h>

#define LOG_TAG "FommEngine"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global FommEngine instance
static FommEngine* gFommEngine = nullptr;

// Helper function: Convert jstring to std::string
std::string jstringToString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string str(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return str;
}

// Helper function: Convert Java byte array to void*
void* byteArrayToVoidPtr(JNIEnv* env, jbyteArray array) {
    if (!array) return nullptr;
    jsize len = env->GetArrayLength(array);
    jbyte* bytes = env->GetByteArrayElements(array, nullptr);
    return static_cast<void*>(bytes);
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

// Convert bitmap pixels to tensor (RGBA to RGB + normalize)
std::vector<float> FommEngine::bitmapToTensor(void* pixels, int width, int height) {
    std::vector<float> tensor(BATCH_SIZE * CHANNELS * TARGET_SIZE * TARGET_SIZE, 0.0f);
    if (!pixels) {
        LOGE("bitmapToTensor: pixels is null");
        return tensor;
    }

    uint32_t* rgbaPixels = static_cast<uint32_t*>(pixels);
    for (int y = 0; y < std::min(height, static_cast<int>(TARGET_SIZE)); ++y) {
        for (int x = 0; x < std::min(width, static_cast<int>(TARGET_SIZE)); ++x) {
            int srcIdx = y * width + x;
            int dstIdx = (y * TARGET_SIZE + x) * CHANNELS;

            uint32_t pixel = rgbaPixels[srcIdx];
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

// Convert tensor to bitmap pixels (RGB to RGBA)
void FommEngine::tensorToBitmap(const float* tensorData, void* outPixels, int width, int height) {
    if (!outPixels || !tensorData) {
        LOGE("tensorToBitmap: outPixels or tensorData is null");
        return;
    }

    uint32_t* rgbaPixels = static_cast<uint32_t*>(outPixels);
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
            rgbaPixels[dstIdx] = (0xFF << 24) | (r8 << 16) | (g8 << 8) | b8;
        }
    }
}

// Extract keypoints from input tensor
FommEngine::Keypoints FommEngine::extractKeypoints(const std::vector<float>& inputTensor) {
    Keypoints keypoints;
    try {
        auto kpInputNames = getInputOutputNames(*kpSession, true);
        auto kpOutputNames = getInputOutputNames(*kpSession, false);

        if (kpInputNames.empty() || kpOutputNames.empty()) {
            LOGE("No input/output names found for kpSession");
            return keypoints;
        }

        auto kpInputNamesPtr = stringVecToCharPtrVec(kpInputNames);
        auto kpOutputNamesPtr = stringVecToCharPtrVec(kpOutputNames);

        std::vector<int64_t> inputShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensorValue = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(inputTensor.data()), inputTensor.size(), inputShape.data(), inputShape.size());

        std::vector<float> kpOutput(BATCH_SIZE * TARGET_SIZE * TARGET_SIZE * 2);
        std::vector<float> jacOutput(BATCH_SIZE * TARGET_SIZE * TARGET_SIZE * 2);
        std::vector<int64_t> kpShape = {BATCH_SIZE, TARGET_SIZE, TARGET_SIZE, 2};
        std::vector<int64_t> jacShape = {BATCH_SIZE, TARGET_SIZE, TARGET_SIZE, 2};

        Ort::Value kpOutputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, kpOutput.data(), kpOutput.size(), kpShape.data(), kpShape.size());
        Ort::Value jacOutputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, jacOutput.data(), jacOutput.size(), jacShape.data(), jacShape.size());

        Ort::Value outputTensors[] = {
            std::move(kpOutputTensor),
            std::move(jacOutputTensor)
        };
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
        LOGD("Processing frame: width=%d, height=%d", width, height);

        std::vector<float> sourceTensor = bitmapToTensor(sourcePixels, width, height);
        std::vector<float> drivingTensor = bitmapToTensor(drivingPixels, width, height);

        Keypoints sourceKeypoints = extractKeypoints(sourceTensor);
        Keypoints drivingKeypoints = extractKeypoints(drivingTensor);

        auto genInputNames = getInputOutputNames(*genSession, true);
        auto genOutputNames = getInputOutputNames(*genSession, false);

        if (genInputNames.empty() || genOutputNames.empty()) {
            LOGE("No input/output names found for genSession");
            return false;
        }

        auto genInputNamesPtr = stringVecToCharPtrVec(genInputNames);
        auto genOutputNamesPtr = stringVecToCharPtrVec(genOutputNames);

        std::vector<int64_t> inputShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensorValue = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(sourceTensor.data()), sourceTensor.size(), inputShape.data(), inputShape.size());

        std::vector<float> outputTensor(BATCH_SIZE * CHANNELS * TARGET_SIZE * TARGET_SIZE);
        std::vector<int64_t> outputShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
        Ort::Value outputTensorValue = Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensor.data(), outputTensor.size(), outputShape.data(), outputShape.size());

        genSession->Run(
            Ort::RunOptions{nullptr},
            genInputNamesPtr.data(), &inputTensorValue, 1,
            genOutputNamesPtr.data(), &outputTensorValue, 1
        );

        tensorToBitmap(outputTensor.data(), outputPixels, width, height);
        LOGD("Frame processed successfully");
        return true;
    } catch (const std::exception& e) {
        LOGE("Exception in processFrame: %s", e.what());
        return false;
    }
}

// JNI: Initialize FommEngine
extern "C" JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_MotionEngine_initEngine(
    JNIEnv* env,
    jobject /* this */,
    jstring kpModelPath,
    jstring genModelPath
) {
    try {
        std::string kpPath = jstringToString(env, kpModelPath);
        std::string genPath = jstringToString(env, genModelPath);
        gFommEngine = new FommEngine();
        return gFommEngine->initialize(kpPath, genPath);
    } catch (const std::exception& e) {
        LOGE("JNI initialize failed: %s", e.what());
        return JNI_FALSE;
    }
}

// JNI: Process a frame
extern "C" JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_MotionEngine_processFrame(
    JNIEnv* env,
    jobject /* this */,
    jbyteArray sourcePixels,
    jbyteArray drivingPixels,
    jbyteArray outputPixels,
    jint width,
    jint height
) {
    if (!gFommEngine) {
        LOGE("FommEngine not initialized");
        return JNI_FALSE;
    }

    void* srcPtr = byteArrayToVoidPtr(env, sourcePixels);
    void* drvPtr = byteArrayToVoidPtr(env, drivingPixels);
    void* outPtr = byteArrayToVoidPtr(env, outputPixels);

    bool result = gFommEngine->processFrame(srcPtr, drvPtr, outPtr, width, height);

    if (sourcePixels) env->ReleaseByteArrayElements(sourcePixels, static_cast<jbyte*>(srcPtr), JNI_ABORT);
    if (drivingPixels) env->ReleaseByteArrayElements(drivingPixels, static_cast<jbyte*>(drvPtr), JNI_ABORT);
    if (outputPixels) env->ReleaseByteArrayElements(outputPixels, static_cast<jbyte*>(outPtr), 0);

    return result ? JNI_TRUE : JNI_FALSE;
}

// JNI: Release FommEngine
extern "C" JNIEXPORT void JNICALL
Java_com_motionforge_app_MotionEngine_releaseEngine(
    JNIEnv* env,
    jobject /* this */
) {
    if (gFommEngine) {
        delete gFommEngine;
        gFommEngine = nullptr;
    }
}