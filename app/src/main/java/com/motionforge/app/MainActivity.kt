package com.motionforge.app

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
// Import these for Coroutines and Lifecycle
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    private lateinit var motionEngine: FommEngineWrapper
    private lateinit var outputView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        outputView = ImageView(this)
        outputView.scaleType = ImageView.ScaleType.FIT_CENTER
        setContentView(outputView)

        motionEngine = FommEngineWrapper()

        // Launched within the Activity's lifecycleScope
        lifecycleScope.launch(Dispatchers.Default) {
            val models = copyAssetsToFilesDir()
            if (models != null && motionEngine.initialize(models.first, models.second)) {
                processTestFrame()
            } else {
                Log.e("MainActivity", "Engine failed to initialize.")
            }
        }
    }

    private fun copyAssetsToFilesDir(): Pair<String, String>? {
        val detectorName = "FOMMDetector.onnx"
        val generatorName = "FOMMGenerator.onnx"
        
        return try {
            val detectorFile = File(filesDir, detectorName)
            val generatorFile = File(filesDir, generatorName)
            
            if (!detectorFile.exists()) assets.open(detectorName).use { it.copyTo(FileOutputStream(detectorFile)) }
            if (!generatorFile.exists()) assets.open(generatorName).use { it.copyTo(FileOutputStream(generatorFile)) }
            
            Pair(detectorFile.absolutePath, generatorFile.absolutePath)
        } catch (e: Exception) {
            null
        }
    }

    private suspend fun processTestFrame() {
        val width = 256
        val height = 256
        val outBytes = ByteArray(width * height * 4)
        
        val success = motionEngine.processFrame(ByteArray(width * height * 4), ByteArray(width * height * 4), outBytes, width, height)
        
        if (success) {
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val ints = IntArray(width * height)
            for (i in ints.indices) {
                ints[i] = ((outBytes[i * 4].toInt() and 0xFF) shl 24) or
                          ((outBytes[i * 4 + 1].toInt() and 0xFF) shl 16) or
                          ((outBytes[i * 4 + 2].toInt() and 0xFF) shl 8) or
                          (outBytes[i * 4 + 3].toInt() and 0xFF)
            }
            bitmap.setPixels(ints, 0, width, 0, 0, width, height)
            
            withContext(Dispatchers.Main) {
                outputView.setImageBitmap(bitmap)
            }
        }
    }
}
