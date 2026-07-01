// MainActivity.kt
package com.motionforge.app

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    private lateinit var fommEngineWrapper: FommEngineWrapper
    private lateinit var sourceBitmap: Bitmap
    private lateinit var drivingBitmap: Bitmap
    private lateinit var outputBitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        fommEngineWrapper = FommEngineWrapper()
        initializeFommEngine()
        loadTestBitmaps()
        processTestFrame()
    }

    private fun initializeFommEngine() {
        try {
            // Replace with actual paths to your ONNX models
            val kpModelPath = "$filesDir/kp_model.onnx"
            val genModelPath = "$filesDir/gen_model.onnx"
            if (!fommEngineWrapper.initialize(kpModelPath, genModelPath)) {
                Log.e("MainActivity", "Failed to initialize FommEngine")
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Exception in initializeFommEngine: ${e.message}")
        }
    }

    private fun loadTestBitmaps() {
        // Load or create test bitmaps (replace with your actual bitmaps)
        sourceBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        drivingBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        outputBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
    }

    private fun processTestFrame() {
        try {
            // Extract pixels from bitmaps
            val sourcePixels = IntArray(256 * 256)
            val drivingPixels = IntArray(256 * 256)
            val outputPixels = IntArray(256 * 256)

            sourceBitmap.getPixels(sourcePixels, 0, 256, 0, 0, 256, 256)
            drivingBitmap.getPixels(drivingPixels, 0, 256, 0, 0, 256, 256)

            // Convert IntArray to ByteArray (RGBA to bytes)
            val sourceBytes = intArrayToByteArray(sourcePixels)
            val drivingBytes = intArrayToByteArray(drivingPixels)
            val outputBytes = ByteArray(256 * 256 * 4) // RGBA

            // Process frame
            val result = fommEngineWrapper.processFrame(
                sourceBytes,
                drivingBytes,
                outputBytes,
                256,
                256
            )

            if (result) {
                // Convert output bytes back to IntArray
                val outputInts = byteArrayToIntArray(outputBytes)
                outputBitmap.setPixels(outputInts, 0, 256, 0, 0, 256, 256)
                Log.d("MainActivity", "Frame processed successfully")
            } else {
                Log.e("MainActivity", "Failed to process frame")
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Exception in processTestFrame: ${e.message}")
        }
    }

    // Helper: Convert IntArray (RGBA) to ByteArray
    private fun intArrayToByteArray(ints: IntArray): ByteArray {
        val bytes = ByteArray(ints.size * 4)
        for (i in ints.indices) {
            val pixel = ints[i]
            bytes[i * 4] = (pixel shr 16 and 0xFF).toByte() // R
            bytes[i * 4 + 1] = (pixel shr 8 and 0xFF).toByte() // G
            bytes[i * 4 + 2] = (pixel and 0xFF).toByte() // B
            bytes[i * 4 + 3] = (pixel shr 24 and 0xFF).toByte() // A
        }
        return bytes
    }

    // Helper: Convert ByteArray (RGBA) to IntArray
    private fun byteArrayToIntArray(bytes: ByteArray): IntArray {
        val ints = IntArray(bytes.size / 4)
        for (i in ints.indices) {
            val r = bytes[i * 4].toInt() and 0xFF
            val g = bytes[i * 4 + 1].toInt() and 0xFF
            val b = bytes[i * 4 + 2].toInt() and 0xFF
            val a = bytes[i * 4 + 3].toInt() and 0xFF
            ints[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }
        return ints
    }
}