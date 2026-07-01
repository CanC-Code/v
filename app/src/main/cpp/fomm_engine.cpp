// fomm_engine.cpp
#include "fomm_engine.h"
#include <android/log.h>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <utility>
#include <jni.h>

#define LOG_TAG "FommEngine"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global FommEngine instance (simplified for example)
static FommEngine* gFommEngine = nullptr;

// JNI Helper: Convert jstring to std::string
std::string jstringToString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string str(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return str;
}

// JNI Helper: Convert Java byte array to void*
void* byteArrayToVoidPtr(JNIEnv* env, jbyteArray array) {
    if (!array) return nullptr;
    jsize len = env->GetArrayLength(array);
    jbyte* bytes = env->GetByteArrayElements(array, nullptr);
    return static_cast<void*>(bytes);
}

// JNI Helper: Convert void* to Java byte array
jbyteArray voidPtrToByteArray(JNIEnv* env, void* data, int size) {
    if (!data) return nullptr;
    jbyteArray array = env->NewByteArray(size);
    env->SetByteArrayRegion(array, 0, size, static_cast<jbyte*>(data));
    return array;
}

// JNI: Initialize FommEngine
extern "C" JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_FommEngineWrapper_initialize(
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
        LOGE("JNI initialize failed: %s", e.what());
        return JNI_FALSE;
    }
}

// JNI: Process a frame
extern "C" JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_FommEngineWrapper_processFrame(
    JNIEnv* env,
    jobject /* this */,
    jbyteArray sourcePixels,
    jbyteArray drivingPixels,
    jbyteArray outputPixels,
    jint width,
    jint height
) {
    if (!gFommEngine) {
        LOGE("FommEngine not initialized");
        return JNI_FALSE;
    }

    void* srcPtr = byteArrayToVoidPtr(env, sourcePixels);
    void* drvPtr = byteArrayToVoidPtr(env, drivingPixels);
    void* outPtr = byteArrayToVoidPtr(env, outputPixels);

    bool result = gFommEngine->processFrame(srcPtr, drvPtr, outPtr, width, height);

    // Release Java arrays
    if (sourcePixels) env->ReleaseByteArrayElements(sourcePixels, static_cast<jbyte*>(srcPtr), JNI_ABORT);
    if (drivingPixels) env->ReleaseByteArrayElements(drivingPixels, static_cast<jbyte*>(drvPtr), JNI_ABORT);
    if (outputPixels) env->ReleaseByteArrayElements(outputPixels, static_cast<jbyte*>(outPtr), 0);

    return result ? JNI_TRUE : JNI_FALSE;
}

// Rest of your existing FommEngine implementation...
// (Keep all the existing methods: constructor, destructor, bitmapToTensor, tensorToBitmap, etc.)