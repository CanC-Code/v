#include <jni.h>
#include <string>
#include <android/log.h>
#include "fomm_engine.h"

// Expose global engine instance initialized in fomm_engine.cpp
extern FommEngine* gFommEngine;

// Forward declare the helper functions that were implemented in fomm_engine.cpp
// but missing from the header. This resolves the 'undeclared identifier' errors.
extern std::string jstringToString(JNIEnv* env, jstring jstr);
extern void* byteArrayToVoidPtr(JNIEnv* env, jbyteArray array);

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_MainActivity_initializeEngine(JNIEnv* env, jobject /* this */, jstring kpModelPath, jstring genModelPath) {
    try {
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
Java_com_motionforge_app_MainActivity_processFrameNative(JNIEnv* env, jobject /* this */, jbyteArray sourcePixels, jbyteArray drivingPixels, jbyteArray outputPixels, jint width, jint height) {
    if (!gFommEngine) {
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "FommEngine not initialized");
        return JNI_FALSE;
    }

    void* srcPtr = byteArrayToVoidPtr(env, sourcePixels);
    void* drvPtr = byteArrayToVoidPtr(env, drivingPixels);
    void* outPtr = byteArrayToVoidPtr(env, outputPixels);

    bool success = gFommEngine->processFrame(srcPtr, drvPtr, outPtr, width, height);
    return success ? JNI_TRUE : JNI_FALSE;
}
