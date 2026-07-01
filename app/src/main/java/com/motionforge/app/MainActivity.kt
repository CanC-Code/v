// MainActivity.kt
package com.motionforge.app

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    private lateinit var motionEngine: MotionEngine
    private lateinit var sourceBitmap: Bitmap
    private lateinit var drivingBitmap: Bitmap
    private lateinit var outputBitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        motionEngine = MotionEngine()
        initializeEngine()
        loadTestBitmaps()
        processTestFrame()
    }

    private fun initializeEngine() {
        try {
            val kpModelPath = "$filesDir/kp_model.onnx"
            val genModelPath = "$filesDir/gen_model.onnx"
            if (!motionEngine.initEngine(kpModelPath, genModelPath)) {
                Log.e("MainActivity", "Failed to initialize MotionEngine")
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Exception in initializeEngine: ${e.message}")
        }
    }

    private fun loadTestBitmaps() {
        sourceBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        drivingBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        outputBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
    }

    private fun processTestFrame() {
        try {
            val sourcePixels = IntArray(256 * 256)
            val drivingPixels = IntArray(256 * 256)
            val outputPixels = IntArray(256 * 256)

            sourceBitmap.getPixels(sourcePixels, 0, 256, 0, 0, 256, 256)
            drivingBitmap.getPixels(drivingPixels, 0, 256, 0, 0, 256, 256)

            val sourceBytes = intArrayToByteArray(sourcePixels)
            val drivingBytes = intArrayToByteArray(drivingPixels)
            val outputBytes = ByteArray(256 * 256 * 4)

            val result = motionEngine.processFrame(
                sourceBytes,
                drivingBytes,
                outputBytes,
                256,
                256
            )

            if (result) {
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

    override fun onDestroy() {
        super.onDestroy()
        motionEngine.releaseEngine()
    }
}