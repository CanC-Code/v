#include <jni.h>
#include <string>
#include <android/log.h>
#include "fomm_engine.h"

// Expose global engine instance initialized in fomm_engine.cpp
extern FommEngine* gFommEngine;

// Forward declare the string helper implemented in fomm_engine.cpp
extern std::string jstringToString(JNIEnv* env, jstring jstr);

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_FommEngineWrapper_initialize(JNIEnv* env, jobject /* this */, jstring kpModelPath, jstring genModelPath) {
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
Java_com_motionforge_app_FommEngineWrapper_processFrame(JNIEnv* env, jobject /* this */, jbyteArray sourcePixels, jbyteArray drivingPixels, jbyteArray outputPixels, jint width, jint height) {
    if (!gFommEngine) {
        __android_log_print(ANDROID_LOG_ERROR, "native_bridge", "FommEngine not initialized");
        return JNI_FALSE;
    }

    // Grab array pointers directly from the JVM
    jbyte* srcPtr = env->GetByteArrayElements(sourcePixels, nullptr);
    jbyte* drvPtr = env->GetByteArrayElements(drivingPixels, nullptr);
    jbyte* outPtr = env->GetByteArrayElements(outputPixels, nullptr);

    bool success = gFommEngine->processFrame(srcPtr, drvPtr, outPtr, width, height);

    // REQUIRED: Release the JNI arrays to prevent memory leaks and GC crashes
    // Use JNI_ABORT for read-only inputs to avoid copying memory back, and 0 for output to save the changes.
    env->ReleaseByteArrayElements(sourcePixels, srcPtr, JNI_ABORT);
    env->ReleaseByteArrayElements(drivingPixels, drvPtr, JNI_ABORT);
    env->ReleaseByteArrayElements(outputPixels, outPtr, 0);

    return success ? JNI_TRUE : JNI_FALSE;
}
