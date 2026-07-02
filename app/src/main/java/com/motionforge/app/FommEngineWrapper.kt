package com.example.motionforge

import android.graphics.Bitmap

class FommEngineWrapper {
    external fun initialize(kpModelPath: String, genModelPath: String): Boolean
    
    external fun processFrame(
        sourceBitmap: Bitmap,
        drivingBitmap: Bitmap,
        outputBitmap: Bitmap,
        isFirstFrame: Boolean
    ): Boolean

    companion object {
        init {
            System.loadLibrary("onnxruntime")
            System.loadLibrary("motionforge_engine")
        }
    }
}
