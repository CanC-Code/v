#include "fomm_engine.h"
#include <android/log.h>
#include <cmath>
#include <algorithm>

#define LOG_TAG "MotionForge-Native"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

FommEngine::FommEngine() {
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MotionForge");
}

FommEngine::~FommEngine() = default;

bool FommEngine::initialize(const std::string& kpModelPath, const std::string& genModelPath) {
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Removed explicit NNAPI call. The onnxruntime-android AAR defaults to 
        // XNNPACK (highly optimized CPU execution) automatically. 
        // The NNAPI provider requires <nnapi_provider_factory.h> which is 
        // not exposed in the root include directory of the Maven AAR.

        kpSession = std::make_unique<Ort::Session>(*env, kpModelPath.c_str(), sessionOptions);
        genSession = std::make_unique<Ort::Session>(*env, genModelPath.c_str(), sessionOptions);
        
        LOGI("ONNX Sessions initialized successfully.");
        return true;
    } catch (const Ort::Exception& e) {
        LOGE("Failed to initialize ONNX sessions: %s", e.what());
        return false;
    }
}

// Equivalent to frameToTensor in motion-engine.js
std::vector<float> FommEngine::bitmapToTensor(void* pixels, int width, int height) {
    int spatial_size = width * height;
    std::vector<float> tensorData(CHANNELS * spatial_size);
    uint8_t* rgba = static_cast<uint8_t*>(pixels);

    for (int i = 0; i < spatial_size; ++i) {
        int pixel_idx = i * 4;
        // Normalize 0-255 to 0.0-1.0 and convert Interleaved RGBA to Planar CHW
        tensorData[i]                   = rgba[pixel_idx]     / 255.0f; // R
        tensorData[i + spatial_size]    = rgba[pixel_idx + 1] / 255.0f; // G
        tensorData[i + spatial_size * 2]= rgba[pixel_idx + 2] / 255.0f; // B
    }
    return tensorData;
}

// Equivalent to tensorToImageData in motion-engine.js
void FommEngine::tensorToBitmap(const float* tensorData, void* outPixels, int width, int height) {
    int spatial_size = width * height;
    uint8_t* rgba = static_cast<uint8_t*>(outPixels);

    for (int i = 0; i < spatial_size; ++i) {
        int pixel_idx = i * 4;
        // Convert Planar CHW back to Interleaved RGBA and map 0.0-1.0 to 0-255
        rgba[pixel_idx]     = static_cast<uint8_t>(std::clamp(std::round(tensorData[i] * 255.0f), 0.0f, 255.0f));
        rgba[pixel_idx + 1] = static_cast<uint8_t>(std::clamp(std::round(tensorData[i + spatial_size] * 255.0f), 0.0f, 255.0f));
        rgba[pixel_idx + 2] = static_cast<uint8_t>(std::clamp(std::round(tensorData[i + spatial_size * 2] * 255.0f), 0.0f, 255.0f));
        rgba[pixel_idx + 3] = 255; // Alpha channel fully opaque
    }
}

FommEngine::Keypoints FommEngine::extractKeypoints(const std::vector<float>& inputTensor) {
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> inputShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
    
    Ort::Value inputOrtTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputTensor.data()), inputTensor.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = {"image"};
    const char* outputNames[] = {"kp", "jac"};

    auto outputTensors = kpSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputOrtTensor, 1, outputNames, 2);

    Keypoints result;
    
    // Extract KP Output
    float* kpData = outputTensors[0].GetTensorMutableData<float>();
    auto kpInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    result.kp_shape = kpInfo.GetShape();
    result.kp.assign(kpData, kpData + kpInfo.GetElementCount());

    // Extract Jacobian Output
    float* jacData = outputTensors[1].GetTensorMutableData<float>();
    auto jacInfo = outputTensors[1].GetTensorTypeAndShapeInfo();
    result.jac_shape = jacInfo.GetShape();
    result.jac.assign(jacData, jacData + jacInfo.GetElementCount());

    return result;
}

bool FommEngine::processFrame(void* sourcePixels, void* drivingPixels, void* outputPixels, int width, int height) {
    if (!kpSession || !genSession) {
        LOGE("Engine not initialized.");
        return false;
    }

    try {
        // 1. Prepare Inputs
        std::vector<float> sourceTensor = bitmapToTensor(sourcePixels, width, height);
        std::vector<float> drivingTensor = bitmapToTensor(drivingPixels, width, height);

        // 2. Extract Keypoints
        Keypoints sourceKps = extractKeypoints(sourceTensor);
        Keypoints drivingKps = extractKeypoints(drivingTensor);

        // 3. Prepare Generator Inputs
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> imageShape = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};

        Ort::Value srcImageVal = Ort::Value::CreateTensor<float>(memoryInfo, sourceTensor.data(), sourceTensor.size(), imageShape.data(), imageShape.size());
        Ort::Value srcKpVal = Ort::Value::CreateTensor<float>(memoryInfo, sourceKps.kp.data(), sourceKps.kp.size(), sourceKps.kp_shape.data(), sourceKps.kp_shape.size());
        Ort::Value srcJacVal = Ort::Value::CreateTensor<float>(memoryInfo, sourceKps.jac.data(), sourceKps.jac.size(), sourceKps.jac_shape.data(), sourceKps.jac_shape.size());
        Ort::Value drvKpVal = Ort::Value::CreateTensor<float>(memoryInfo, drivingKps.kp.data(), drivingKps.kp.size(), drivingKps.kp_shape.data(), drivingKps.kp_shape.size());
        Ort::Value drvJacVal = Ort::Value::CreateTensor<float>(memoryInfo, drivingKps.jac.data(), drivingKps.jac.size(), drivingKps.jac_shape.data(), drivingKps.jac_shape.size());

        std::vector<const char*> inputNames = {"source_image", "source_keypoints", "source_jacobian", "driving_keypoints", "driving_jacobian"};
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(srcImageVal));
        inputTensors.push_back(std::move(srcKpVal));
        inputTensors.push_back(std::move(srcJacVal));
        inputTensors.push_back(std::move(drvKpVal));
        inputTensors.push_back(std::move(drvJacVal));

        const char* outputNames[] = {"output_image"}; 

        // 4. Run Generation
        auto outputTensors = genSession->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), inputTensors.size(), outputNames, 1);

        // 5. Retrieve output and convert back to Bitmap
        float* outData = outputTensors[0].GetTensorMutableData<float>();
        tensorToBitmap(outData, outputPixels, width, height);

        return true;

    } catch (const Ort::Exception& e) {
        LOGE("ONNX Error during processFrame: %s", e.what());
        return false;
    }
}
