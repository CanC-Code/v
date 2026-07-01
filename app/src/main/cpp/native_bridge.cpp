#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include "fomm_engine.h"

// Global pointer to hold our engine instance
FommEngine* engine = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_MotionEngine_initEngine(JNIEnv* env, jobject /* this */, jstring kpPath, jstring genPath) {
    if (engine == nullptr) {
        engine = new FommEngine();
    }
    
    const char* nativeKpPath = env->GetStringUTFChars(kpPath, nullptr);
    const char* nativeGenPath = env->GetStringUTFChars(genPath, nullptr);

    bool result = engine->initialize(nativeKpPath, nativeGenPath);

    env->ReleaseStringUTFChars(kpPath, nativeKpPath);
    env->ReleaseStringUTFChars(genPath, nativeGenPath);

    return result ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_motionforge_app_MotionEngine_processFrame(JNIEnv* env, jobject /* this */, jobject sourceBitmap, jobject drivingBitmap, jobject outputBitmap) {
    if (engine == nullptr) {
        return JNI_FALSE;
    }

    AndroidBitmapInfo info;
    void* sourcePixels;
    void* drivingPixels;
    void* outputPixels;

    // Retrieve bitmap info to ensure it matches expectations (256x256 RGBA_8888)
    if (AndroidBitmap_getInfo(env, sourceBitmap, &info) < 0) return JNI_FALSE;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return JNI_FALSE;

    // Lock pixel buffers for direct native memory access
    if (AndroidBitmap_lockPixels(env, sourceBitmap, &sourcePixels) < 0) return JNI_FALSE;
    if (AndroidBitmap_lockPixels(env, drivingBitmap, &drivingPixels) < 0) {
        AndroidBitmap_unlockPixels(env, sourceBitmap);
        return JNI_FALSE;
    }
    if (AndroidBitmap_lockPixels(env, outputBitmap, &outputPixels) < 0) {
        AndroidBitmap_unlockPixels(env, drivingBitmap);
        AndroidBitmap_unlockPixels(env, sourceBitmap);
        return JNI_FALSE;
    }

    // Process the frame using the locked pointers
    bool success = engine->processFrame(sourcePixels, drivingPixels, outputPixels, info.width, info.height);

    // Release buffers
    AndroidBitmap_unlockPixels(env, outputBitmap);
    AndroidBitmap_unlockPixels(env, drivingBitmap);
    AndroidBitmap_unlockPixels(env, sourceBitmap);

    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_motionforge_app_MotionEngine_releaseEngine(JNIEnv* env, jobject /* this */) {
    if (engine != nullptr) {
        delete engine;
        engine = nullptr;
    }
}
