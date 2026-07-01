// FommEngineWrapper.kt
package com.motionforge.app

class FommEngineWrapper {
    external fun initialize(kpModelPath: String, genModelPath: String): Boolean
    external fun processFrame(
        sourcePixels: ByteArray,
        drivingPixels: ByteArray,
        outputPixels: ByteArray,
        width: Int,
        height: Int
    ): Boolean

    companion object {
        init {
            System.loadLibrary("motionforge_engine")
        }
    }
}