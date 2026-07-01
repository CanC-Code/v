package com.motionforge.app

import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var engine: MotionEngine
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        engine = MotionEngine()

        // 1. Initialize Engine with absolute paths to the extracted assets
        val kpPath = File(filesDir.parentFile, "app_assets/FOMMDetector.onnx").absolutePath
        val genPath = File(filesDir.parentFile, "app_assets/FOMMGenerator.onnx").absolutePath
        
        // Ensure you copy models from /assets to /files directory if needed by C++ file system
        engine.initEngine(kpPath, genPath)

        findViewById<Button>(R.id.btnStart).setOnClickListener {
            // Placeholder: Invoke processFrame() here after camera frames are captured
            // engine.processFrame(sourceBitmap, drivingBitmap, outputBitmap)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.releaseEngine()
    }
}
