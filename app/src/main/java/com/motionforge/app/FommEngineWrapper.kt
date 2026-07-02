package com.example.motionforge

class FommEngineWrapper {
    external fun initialize(kpModelPath: String, genModelPath: String): Boolean
    
    external fun processVideo(
        sourceImagePath: String,
        drivingVideoPath: String,
        outputPath: String
    ): Boolean

    companion object {
        init {
            // CRITICAL FIX: The dependent library must be loaded into memory 
            // before the Android dynamic linker attempts to load motionforge_engine
            System.loadLibrary("onnxruntime")
            System.loadLibrary("motionforge_engine")
        }
    }
}
