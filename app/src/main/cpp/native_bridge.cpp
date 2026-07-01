#include <jni.h>
#include <string>
#include <android/log.h>
#include "fomm_engine.h"

// Expose global engine instance initialized in fomm_engine.cpp
extern FommEngine* gFommEngine;

// jstringToString is now declared in fomm_engine.h (included above),
// no local forward declaration needed.

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
Java_com_motionforge_app_FommEngineWrapper_processFrame(JNIEnv* env, jobject /* this */, jbyteArray sourcePixels, jbyteArray drivingPixels, jbyteArray outputPixels, jint width, jint height) {
    if (!gFommEngine) {
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "FommEngine not initialized");
        return JNI_FALSE;
    }

    if (!sourcePixels || !drivingPixels || !outputPixels) {
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "Null array provided to JNI");
        return JNI_FALSE;
    }

    jbyte* srcPtr = env->GetByteArrayElements(sourcePixels, nullptr);
    jbyte* drvPtr = env->GetByteArrayElements(drivingPixels, nullptr);
    jbyte* outPtr = env->GetByteArrayElements(outputPixels, nullptr);

    if (!srcPtr || !drvPtr || !outPtr) {
        if (srcPtr) env->ReleaseByteArrayElements(sourcePixels, srcPtr, JNI_ABORT);
        if (drvPtr) env->ReleaseByteArrayElements(drivingPixels, drvPtr, JNI_ABORT);
        if (outPtr) env->ReleaseByteArrayElements(outputPixels, outPtr, JNI_ABORT);
        return JNI_FALSE;
    }

    bool success = gFommEngine->processFrame(srcPtr, drvPtr, outPtr, width, height);

    env->ReleaseByteArrayElements(sourcePixels, srcPtr, JNI_ABORT);
    env->ReleaseByteArrayElements(drivingPixels, drvPtr, JNI_ABORT);
    env->ReleaseByteArrayElements(outputPixels, outPtr, 0);

    return success ? JNI_TRUE : JNI_FALSE;
}