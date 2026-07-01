// MainActivity.kt
package com.motionforge.app

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    private lateinit var motionEngine: FommEngineWrapper
    private lateinit var sourceBitmap: Bitmap
    private lateinit var drivingBitmap: Bitmap
    private lateinit var outputBitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        motionEngine = FommEngineWrapper()
        initializeEngine()
        loadTestBitmaps()
        processTestFrame()
    }
    
    // Extracts both the ONNX models and their required .data weight files
    private fun copyAssetsToFilesDir(): Pair<String, String>? {
        try {
            val assetFiles = assets.list("") ?: return null
            var detectorPath = ""
            var generatorPath = ""
            
            for (file in assetFiles) {
                // Ensure we copy both structure (.onnx) and weights (.data)
                if (file.endsWith(".onnx") || file.endsWith(".data")) {
                    val destFile = File(filesDir, file)
                    if (!destFile.exists()) {
                        assets.open(file).use { inputStream ->
                            FileOutputStream(destFile).use { outputStream ->
                                inputStream.copyTo(outputStream)
                            }
                        }
                    }
                    if (file.endsWith("FOMMDetector.onnx")) {
                        detectorPath = destFile.absolutePath
                    } else if (file.endsWith("FOMMGenerator.onnx")) {
                        generatorPath = destFile.absolutePath
                    }
                }
            }
            
            if (detectorPath.isEmpty() || generatorPath.isEmpty()) {
                Log.e("MainActivity", "Required ONNX models not found in assets.")
                return null
            }
            
            return Pair(detectorPath, generatorPath)
        } catch (e: Exception) {
            Log.e("MainActivity", "Exception copying assets: ${e.message}")
            return null
        }
    }

    private fun initializeEngine() {
        val modelPaths = copyAssetsToFilesDir()
        if (modelPaths == null) {
            Log.e("MainActivity", "Aborting initialization: Models missing.")
            return
        }

        try {
            if (!motionEngine.initialize(modelPaths.first, modelPaths.second)) {
                Log.e("MainActivity", "Failed to initialize FommEngineWrapper")
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
            bytes[i * 4] = (pixel shr 24 and 0xFF).toByte()
            bytes[i * 4 + 1] = (pixel shr 16 and 0xFF).toByte()
            bytes[i * 4 + 2] = (pixel shr 8 and 0xFF).toByte()
            bytes[i * 4 + 3] = (pixel and 0xFF).toByte()
        }
        return bytes
    }

    private fun byteArrayToIntArray(bytes: ByteArray): IntArray {
        val ints = IntArray(bytes.size / 4)
        for (i in ints.indices) {
            ints[i] = (bytes[i * 4].toInt() and 0xFF shl 24) or
                      (bytes[i * 4 + 1].toInt() and 0xFF shl 16) or
                      (bytes[i * 4 + 2].toInt() and 0xFF shl 8) or
                      (bytes[i * 4 + 3].toInt() and 0xFF)
        }
        return ints
    }
}
