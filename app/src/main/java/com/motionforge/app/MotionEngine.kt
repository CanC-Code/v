// MotionEngine.kt
package com.motionforge.app

class MotionEngine {
    external fun initEngine(kpModelPath: String, genModelPath: String): Boolean
    external fun processFrame(
        sourcePixels: ByteArray,
        drivingPixels: ByteArray,
        outputPixels: ByteArray,
        width: Int,
        height: Int
    ): Boolean
    external fun releaseEngine()

    companion object {
        init {
            System.loadLibrary("motionforge_engine")
        }
    }
}