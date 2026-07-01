package com.motionforge.app

import android.graphics.Bitmap

class MotionEngine {
    companion object {
        init {
            // Loads the shared library compiled via CMakeLists.txt
            System.loadLibrary("motionforge_engine")
        }
    }

    // Maps directly to the Java_com_motionforge_app_MotionEngine_* JNI functions
    external fun initEngine(kpPath: String, genPath: String): Boolean
    external fun processFrame(sourceBitmap: Bitmap, drivingBitmap: Bitmap, outputBitmap: Bitmap): Boolean
    external fun releaseEngine()
}
