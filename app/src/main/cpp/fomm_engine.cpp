#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

// Helper function to get input/output names from ONNX model
std::vector<std::string> getInputOutputNames(Ort::Session& session, bool isInput) {
    std::vector<std::string> names;
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numNodes = isInput ? session.GetInputCount() : session.GetOutputCount();
    for (size_t i = 0; i < numNodes; ++i) {
        char* name = isInput ?
            session.GetInputName(i, allocator) :
            session.GetOutputName(i, allocator);
        names.emplace_back(name);
    }
    return names;
}

// Updated processFrame function
bool FommEngine::processFrame(...) {
    try {
        // Dynamically discover input/output names
        auto inputNames = getInputOutputNames(*kpSession, true);
        auto outputNames = getInputOutputNames(*kpSession, false);

        // Use discovered names for input/output tensors
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());
        Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, outputData.data(), outputData.size(), outputShape.data(), outputShape.size());

        // Run session with discovered names
        kpSession->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(), &inputTensor, 1,
            outputNames.data(), &outputTensor, 1
        );

        // Repeat for genSession if needed
        auto genInputNames = getInputOutputNames(*genSession, true);
        auto genOutputNames = getInputOutputNames(*genSession, false);
        genSession->Run(
            Ort::RunOptions{nullptr},
            genInputNames.data(), &inputTensor, 1,
            genOutputNames.data(), &outputTensor, 1
        );

        return true;
    } catch (const std::exception& e) {
        // Log error (recommended: add logging here)
        return false;
    }
}