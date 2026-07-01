// fomm_engine.cpp
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <android/log.h>

#define LOG_TAG "FommEngine"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Minimal FommEngine class definition
class FommEngine {
public:
    static bool processFrame(
        Ort::Session* kpSession,
        Ort::Session* genSession,
        const std::vector<float>& inputData,
        const std::vector<int64_t>& inputShape,
        std::vector<float>& outputData,
        const std::vector<int64_t>& outputShape
    );
};

// Helper function to get input/output names from ONNX model
std::vector<std::string> getInputOutputNames(Ort::Session& session, bool isInput) {
    std::vector<std::string> names;
    size_t numNodes = isInput ? session.GetInputCount() : session.GetOutputCount();
    for (size_t i = 0; i < numNodes; ++i) {
        char* name;
        if (isInput) {
            session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions(), &name);
        } else {
            session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions(), &name);
        }
        names.emplace_back(name);
    }
    return names;
}

bool FommEngine::processFrame(
    Ort::Session* kpSession,
    Ort::Session* genSession,
    const std::vector<float>& inputData,
    const std::vector<int64_t>& inputShape,
    std::vector<float>& outputData,
    const std::vector<int64_t>& outputShape
) {
    try {
        // Dynamically discover input/output names for kpSession
        auto kpInputNames = getInputOutputNames(*kpSession, true);
        auto kpOutputNames = getInputOutputNames(*kpSession, false);

        // Dynamically discover input/output names for genSession
        auto genInputNames = getInputOutputNames(*genSession, true);
        auto genOutputNames = getInputOutputNames(*genSession, false);

        // Create input tensor
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(inputData.data()), inputData.size(), inputShape.data(), inputShape.size());

        // Run kpSession
        Ort::Value kpOutputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, outputData.data(), outputData.size(), outputShape.data(), outputShape.size());
        kpSession->Run(
            Ort::RunOptions{nullptr},
            kpInputNames.data(), &inputTensor, 1,
            kpOutputNames.data(), &kpOutputTensor, 1
        );

        // Run genSession
        Ort::Value genOutputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, outputData.data(), outputData.size(), outputShape.data(), outputShape.size());
        genSession->Run(
            Ort::RunOptions{nullptr},
            genInputNames.data(), &inputTensor, 1,
            genOutputNames.data(), &genOutputTensor, 1
        );

        return true;
    } catch (const std::exception& e) {
        LOGE("Exception in processFrame: %s", e.what());
        return false;
    }
}