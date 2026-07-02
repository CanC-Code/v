#pragma once

#include <jni.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

std::string jstringToString(JNIEnv* env, jstring jstr);

class FommEngine {
public:
    FommEngine();
    ~FommEngine();

    bool initialize(const std::string& kpModelPath, const std::string& genModelPath);
    bool processFrame(JNIEnv* env, jobject sourceBitmap, jobject drivingBitmap, jobject outputBitmap, bool isFirstFrame);

private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> kpSession;
    std::unique_ptr<Ort::Session> genSession;

    const int64_t BATCH_SIZE = 1;
    const int64_t CHANNELS = 3;
    const int64_t TARGET_SIZE = 256;

    std::vector<float> cachedSourceKp;
    std::vector<float> cachedInitialDrivingKp;
    std::vector<float> sourceImageBuffer;

    std::vector<float> extractKeypoints(const std::vector<float>& inputFrame);
    std::vector<float> generateFrame(const std::vector<float>& sourceFrame, 
                                     const std::vector<float>& kpSource, 
                                     const std::vector<float>& kpDriving, 
                                     const std::vector<float>& kpDrivingInitial);
};
