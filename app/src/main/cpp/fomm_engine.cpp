#include "fomm_engine.h"
#include <android/log.h>
#include <vector>
#include <cstring>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "FommEngine", __VA_ARGS__)

FommEngine* gFommEngine = nullptr;

FommEngine::FommEngine() 
    : env(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "FommEngine")),
      allocator(Ort::AllocatorWithDefaultOptions()) {}

bool FommEngine::initialize(const std::string& kpPath, const std::string& genPath) {
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(2);
        kpSession = std::make_unique<Ort::Session>(*env, kpPath.c_str(), options);
        genSession = std::make_unique<Ort::Session>(*env, genPath.c_str(), options);
        return true;
    } catch (...) {
        return false;
    }
}

bool FommEngine::processFrame(void* src, void* drv, void* out, int w, int h) {
    if (!kpSession || !genSession) return false;

    // Use default RunOptions() to prevent null-ptr segfaults
    Ort::RunOptions runOptions;

    // Create dummy input to satisfy the generator graph
    std::vector<int64_t> shape = {1, 3, 256, 256};
    std::vector<float> dummyInput(1 * 3 * 256 * 256, 0.0f);
    
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, dummyInput.data(), dummyInput.size(), shape.data(), shape.size());

    const char* inputNames[] = {"input"};
    const char* outputNames[] = {"output"};

    try {
        auto outputTensors = genSession->Run(runOptions, inputNames, &inputTensor, 1, outputNames, 1);
        float* data = outputTensors[0].GetTensorMutableData<float>();
        
        // Fill output with a visible test pattern to verify render pipeline
        std::memset(out, 255, w * h * 4); 
        LOGD("Engine Run Successful");
        return true;
    } catch (const std::exception& e) {
        LOGD("Run Error: %s", e.what());
        return false;
    }
}
