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
    return Java_com_motionforge_app_MotionEngine_initEngine(env, nullptr, kpModelPath, genModelPath);
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
    return Java_com_motionforge_app_MotionEngine_processFrame(env, nullptr, sourcePixels, drivingPixels, outputPixels, width, height);
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