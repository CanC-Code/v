// native_bridge.cpp
#include <jni.h>
#include "fomm_engine.h"

// Declare the global FommEngine instance (defined in fomm_engine.cpp)
extern FommEngine* gFommEngine;

// JNI: Initialize Engine
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
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "JNI initialize failed: %s", e.what());
        return JNI_FALSE;
    }
}

// JNI: Process Frame
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
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "FommEngine not initialized");
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

// JNI: Release Engine
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