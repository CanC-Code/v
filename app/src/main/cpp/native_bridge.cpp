#include <jni.h>
#include <string>
#include <android/log.h>
#include "fomm_engine.h"

extern FommEngine* gFommEngine;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_FommEngineWrapper_initialize(JNIEnv* env, jobject /* this */, jstring kpModelPath, jstring genModelPath) {
    try {
        if (!kpModelPath || !genModelPath) return JNI_FALSE;

        std::string kpPath = jstringToString(env, kpModelPath);
        std::string genPath = jstringToString(env, genModelPath);

        if (!gFommEngine) {
            gFommEngine = new FommEngine();
        }
        
        bool success = gFommEngine->initialize(kpPath, genPath);
        return success ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "JNI initialize failed: %s", e.what());
        return JNI_FALSE;
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_FommEngineWrapper_processVideo(JNIEnv* env, jobject /* this */, jstring imagePath, jstring videoPath, jstring outputPath) {
    if (!gFommEngine) {
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "FommEngine not initialized");
        return JNI_FALSE;
    }

    if (!imagePath || !videoPath || !outputPath) {
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "Null path provided to JNI");
        return JNI_FALSE;
    }

    std::string imgPath = jstringToString(env, imagePath);
    std::string vidPath = jstringToString(env, videoPath);
    std::string outPath = jstringToString(env, outputPath);

    bool success = gFommEngine->processVideo(imgPath, vidPath, outPath);

    return success ? JNI_TRUE : JNI_FALSE;
}
