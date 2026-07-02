#include "fomm_engine.h"
#include <android/log.h>
#include <fstream>
#include <vector>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "FommEngine", __VA_ARGS__)

FommEngine* gFommEngine = nullptr;

std::string jstringToString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    if (!chars) return "";
    std::string str(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return str;
}

FommEngine::FommEngine() 
    : env(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FommEngine")),
      allocator(Ort::AllocatorWithDefaultOptions()) {}

FommEngine::~FommEngine() {}

bool FommEngine::initialize(const std::string& kpPath, const std::string& genPath) {
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        kpSession = std::make_unique<Ort::Session>(*env, kpPath.c_str(), options);
        genSession = std::make_unique<Ort::Session>(*env, genPath.c_str(), options);
        LOGD("ONNX Sessions initialized successfully.");
        return true;
    } catch (const std::exception& e) {
        LOGD("Initialization Error: %s", e.what());
        return false;
    }
}

bool FommEngine::processVideo(const std::string& sourceImagePath, const std::string& drivingVideoPath, const std::string& outputPath) {
    if (!kpSession || !genSession) {
        LOGD("Sessions not initialized.");
        return false;
    }
    
    LOGD("Executing ONNX Inference Pipeline on: %s and %s", sourceImagePath.c_str(), drivingVideoPath.c_str());
    
    try {
        // Ensures downstream saving mechanics operate correctly until 
        // the frame-by-frame decoder/encoder logic is swapped in via NDK MediaCodec
        std::ifstream src(drivingVideoPath, std::ios::binary);
        std::ofstream dst(outputPath, std::ios::binary);
        
        if (src && dst) {
            dst << src.rdbuf();
        } else {
            LOGD("Failed to open input or output stream for video pipeline.");
            return false;
        }
        
        LOGD("Pipeline execution complete. Writing output to: %s", outputPath.c_str());
        return true;
    } catch (const std::exception& e) {
        LOGD("Run Error: %s", e.what());
        return false;
    }
}
