package com.motionforge.app

class FommEngineWrapper {
    external fun initialize(kpModelPath: String, genModelPath: String): Boolean
    
    external fun processVideo(
        sourceImagePath: String,
        drivingVideoPath: String,
        outputPath: String
    ): Boolean

    companion object {
        init {
            System.loadLibrary("motionforge_engine")
        }
    }
}
