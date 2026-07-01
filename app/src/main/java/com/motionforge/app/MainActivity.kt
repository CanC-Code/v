package com.motionforge.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.view.TextureView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var engine: MotionEngine
    private lateinit var outputView: TextureView
    private val executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        outputView = findViewById(R.id.outputView)
        engine = MotionEngine()

        initModels()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 10)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(findViewById<PreviewView>(R.id.viewFinder).surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(256, 256))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(executor) { image ->
                processFrame(image)
            }

            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(image: ImageProxy) {
        val bitmap = image.toBitmap() // Extension function below
        val outputBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)

        // Native Inference Call
        engine.processFrame(bitmap, bitmap, outputBitmap)

        // Render to UI
        val canvas = outputView.lockCanvas()
        if (canvas != null) {
            canvas.drawBitmap(outputBitmap, 0f, 0f, null)
            outputView.unlockCanvasAndPost(canvas)
        }
        image.close()
    }

    // Helper: Convert ImageProxy (YUV) to Bitmap (RGBA)
    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        // ... Standard Android YUV_420_888 to Bitmap conversion logic goes here
        // For production, use RenderScript or a library like 'YuvToRgbConverter'
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888) 
    }

    private fun initModels() {
        val files = listOf("FOMMDetector.onnx", "FOMMGenerator.onnx", "FOMMDetector.data", "FOMMGenerator.data")
        files.forEach { name ->
            val file = File(filesDir, name)
            if (!file.exists()) {
                assets.open(name).use { input -> file.outputStream().use { input.copyTo(it) } }
            }
        }
        engine.initEngine(File(filesDir, "FOMMDetector.onnx").absolutePath, File(filesDir, "FOMMGenerator.onnx").absolutePath)
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    
    override fun onDestroy() {
        super.onDestroy()
        engine.releaseEngine()
        executor.shutdown()
    }
}
