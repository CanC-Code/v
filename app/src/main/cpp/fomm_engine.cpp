#include "fomm_engine.h"
#include <android/log.h>
#include <android/bitmap.h>
#include <algorithm>
#include <string>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "FommEngine", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "FommEngine", __VA_ARGS__)

FommEngine* gFommEngine = nullptr;

std::string jstringToString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    if (!chars) return "";
    std::string str(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return str;
}

FommEngine::FommEngine() : env(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FommEngine")) {}
FommEngine::~FommEngine() {}

bool FommEngine::initialize(const std::string& kpPath, const std::string& genPath) {
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        kpSession = std::make_unique<Ort::Session>(*env, kpPath.c_str(), options);
        genSession = std::make_unique<Ort::Session>(*env, genPath.c_str(), options);
        return true;
    } catch (const std::exception& e) {
        LOGE("Initialization Error: %s", e.what());
        return false;
    }
}

std::vector<float> FommEngine::extractKeypoints(const std::vector<float>& inputFrame) {
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> inputDims = {BATCH_SIZE, CHANNELS, TARGET_SIZE, TARGET_SIZE};
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, 
                                                            const_cast<float*>(inputFrame.data()), 
                                                            inputFrame.size(), 
                                                            inputDims.data(), 
                                                            inputDims.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNamePtr = kpSession->GetInputNameAllocated(0, allocator);
    auto outputNamePtr = kpSession->GetOutputNameAllocated(0, allocator);
    
    const char* inputNames[] = {inputNamePtr.get()};
    const char* outputNames[] = {outputNamePtr.get()};

    auto outputTensors = kpSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    float* floatarr = outputTensors.front().GetTensorMutableData<float>();
    size_t count = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(floatarr, floatarr + count);
}

std::vector<float> FommEngine::generateFrame(const std::vector<float>& sourceFrame, 
                                             const std::vector<float>& kpSource, 
                                             const std::vector<float>& kpDriving, 
                                             const std::vector<float>& kpDrivingInitial) {
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;
    
    size_t inputCount = genSession->GetInputCount();
    
    std::vector<Ort::AllocatedStringPtr> inputNamePtrs;
    std::vector<const char*> inputNames;
    std::vector<Ort::Value> inputTensors;
    std::vector<std::vector<float>> inputBuffers; 
    
    inputBuffers.resize(inputCount);

    for (size_t i = 0; i < inputCount; i++) {
        inputNamePtrs.push_back(genSession->GetInputNameAllocated(i, allocator));
        std::string name = inputNamePtrs.back().get();
        inputNames.push_back(inputNamePtrs.back().get());

        auto typeInfo = genSession->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensorInfo.GetShape();
        
        size_t elementCount = 1;
        for (auto& dim : dims) {
            if (dim <= 0) dim = 1; 
            elementCount *= dim;
        }
        
        inputBuffers[i].resize(elementCount, 0.0f); 

        if (name.find("jacobian") != std::string::npos && dims.size() == 4) {
            int num_kp = dims[1]; 
            for (int k = 0; k < num_kp; k++) {
                inputBuffers[i][k * 4 + 0] = 1.0f;
                inputBuffers[i][k * 4 + 3] = 1.0f; 
            }
        } 
        else if (name == "kp_driving") {
            std::copy(kpDriving.begin(), kpDriving.end(), inputBuffers[i].begin());
        } 
        else if (name == "kp_source") {
            std::copy(kpSource.begin(), kpSource.end(), inputBuffers[i].begin());
        } 
        else if (name == "kp_driving_initial") {
            std::copy(kpDrivingInitial.begin(), kpDrivingInitial.end(), inputBuffers[i].begin());
        } 
        else if (name == "source_image") {
            std::copy(sourceFrame.begin(), sourceFrame.end(), inputBuffers[i].begin());
        } 
        else {
            // Strict heuristics fallback for alternative export names
            if (name.find("initial") != std::string::npos && dims.size() == 3) {
                 std::copy(kpDrivingInitial.begin(), kpDrivingInitial.end(), inputBuffers[i].begin());
            } else if (name.find("driving") != std::string::npos && dims.size() == 3) {
                 std::copy(kpDriving.begin(), kpDriving.end(), inputBuffers[i].begin());
            } else if (name.find("source") != std::string::npos && dims.size() == 3) {
                 std::copy(kpSource.begin(), kpSource.end(), inputBuffers[i].begin());
            } else if (name.find("image") != std::string::npos && dims.size() == 4) {
                 std::copy(sourceFrame.begin(), sourceFrame.end(), inputBuffers[i].begin());
            }
        }

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputBuffers[i].data(), inputBuffers[i].size(), dims.data(), dims.size()
        ));
    }

    auto outputNamePtr = genSession->GetOutputNameAllocated(0, allocator);
    const char* outputNames[] = {outputNamePtr.get()};

    auto outputTensors = genSession->Run(Ort::RunOptions{nullptr}, 
                                         inputNames.data(), inputTensors.data(), inputTensors.size(), 
                                         outputNames, 1);

    float* floatarr = outputTensors.front().GetTensorMutableData<float>();
    size_t count = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(floatarr, floatarr + count);
}

bool FommEngine::processFrame(JNIEnv* env, jobject sourceBitmap, jobject drivingBitmap, jobject outputBitmap, bool isFirstFrame) {
    if (!kpSession || !genSession) return false;

    if (isFirstFrame) {
        cachedSourceKp.clear();
        cachedInitialDrivingKp.clear();
        sourceImageBuffer.assign(BATCH_SIZE * CHANNELS * TARGET_SIZE * TARGET_SIZE, 0.0f);
    }
    
    std::vector<float> drivingBuffer(BATCH_SIZE * CHANNELS * TARGET_SIZE * TARGET_SIZE, 0.0f);
    AndroidBitmapInfo info;
    void* pixels;
    
    if (isFirstFrame) {
        AndroidBitmap_getInfo(env, sourceBitmap, &info);
        AndroidBitmap_lockPixels(env, sourceBitmap, &pixels);
        uint32_t* src = (uint32_t*)pixels;
        for (int i = 0; i < info.width * info.height; ++i) {
            uint32_t c = src[i];
            sourceImageBuffer[i] = (c & 0xFF) / 255.0f;
            sourceImageBuffer[info.width * info.height + i] = ((c >> 8) & 0xFF) / 255.0f;
            sourceImageBuffer[2 * info.width * info.height + i] = ((c >> 16) & 0xFF) / 255.0f;
        }
        AndroidBitmap_unlockPixels(env, sourceBitmap);
        
        cachedSourceKp = extractKeypoints(sourceImageBuffer);
    }

    AndroidBitmap_getInfo(env, drivingBitmap, &info);
    AndroidBitmap_lockPixels(env, drivingBitmap, &pixels);
    uint32_t* drv = (uint32_t*)pixels;
    for (int i = 0; i < info.width * info.height; ++i) {
        uint32_t c = drv[i];
        drivingBuffer[i] = (c & 0xFF) / 255.0f;
        drivingBuffer[info.width * info.height + i] = ((c >> 8) & 0xFF) / 255.0f;
        drivingBuffer[2 * info.width * info.height + i] = ((c >> 16) & 0xFF) / 255.0f;
    }
    AndroidBitmap_unlockPixels(env, drivingBitmap);

    if (isFirstFrame) {
        cachedInitialDrivingKp = extractKeypoints(drivingBuffer);
    }

    std::vector<float> kpDriving = extractKeypoints(drivingBuffer);
    std::vector<float> outputFrame = generateFrame(sourceImageBuffer, cachedSourceKp, kpDriving, cachedInitialDrivingKp);

    AndroidBitmap_getInfo(env, outputBitmap, &info);
    AndroidBitmap_lockPixels(env, outputBitmap, &pixels);
    uint32_t* out = (uint32_t*)pixels;
    for (int i = 0; i < info.width * info.height; ++i) {
        float r = std::clamp(outputFrame[i], 0.0f, 1.0f) * 255.0f;
        float g = std::clamp(outputFrame[info.width * info.height + i], 0.0f, 1.0f) * 255.0f;
        float b = std::clamp(outputFrame[2 * info.width * info.height + i], 0.0f, 1.0f) * 255.0f;
        out[i] = (0xFF << 24) | (((uint32_t)b & 0xFF) << 16) | (((uint32_t)g & 0xFF) << 8) | ((uint32_t)r & 0xFF);
    }
    AndroidBitmap_unlockPixels(env, outputBitmap);

    return true;
}
