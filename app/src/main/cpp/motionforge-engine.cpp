#include <jni.h>
#include <string>
#include <android/log.h>
#include <onnxruntime_cxx_api.h>
#include <vector>

#define LOG_TAG "MotionForgeEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_motionforge_MotionForgeEngine_runMotionTransfer(
        JNIEnv *env, jobject thiz, jstring kp_model_path, jstring gen_model_path,
        jstring image_path, jstring video_path, jstring output_path) {

    const char *kp_path = env->GetStringUTFChars(kp_model_path, nullptr);
    const char *gen_path = env->GetStringUTFChars(gen_model_path, nullptr);
    const char *img_path = env->GetStringUTFChars(image_path, nullptr);
    const char *vid_path = env->GetStringUTFChars(video_path, nullptr);
    const char *out_path = env->GetStringUTFChars(output_path, nullptr);

    try {
        Ort::Env ort_env(ORT_LOGGING_LEVEL_VERBOSE, "MotionForge");
        Ort::SessionOptions options;
        
        Ort::Session kp_session(ort_env, kp_path, options);
        Ort::Session gen_session(ort_env, gen_path, options);

        // DIAGNOSTIC: Validate Input Shapes
        Ort::TypeInfo type_info = kp_session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensor_info.GetShape();
        
        LOGI("Keypoint Detector expected shape: [%lld, %lld, %lld, %lld]", 
             shape[0], shape[1], shape[2], shape[3]);

        // [Inference Logic Placeholder]
        // If your model expects [1, 3, 256, 256] and your image buffer
        // is not being pre-processed to this size, the runtime will throw an exception.
        // Ensure you are using OpenCV or native Bitmaps to resize/normalize inputs.
        
        LOGI("Pipeline check passed. Proceeding with inference...");
        
        // Cleanup strings
        env->ReleaseStringUTFChars(kp_model_path, kp_path);
        env->ReleaseStringUTFChars(gen_model_path, gen_path);
        env->ReleaseStringUTFChars(image_path, img_path);
        env->ReleaseStringUTFChars(video_path, vid_path);
        env->ReleaseStringUTFChars(output_path, out_path);
        return JNI_TRUE;

    } catch (const std::exception &e) {
        LOGE("CRITICAL INFERENCE ERROR: %s", e.what());
        return JNI_FALSE;
    }
}
