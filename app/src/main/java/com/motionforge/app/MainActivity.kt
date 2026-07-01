package com.motionforge.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.*
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var engine: MotionEngine
    private var sourceBitmap: Bitmap? = null
    private var drivingBitmap: Bitmap? = null

    private val pickSource = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { 
            val bmp = uriToBitmap(it)
            sourceBitmap = bmp
            findViewById<ImageView>(R.id.imgSource).setImageBitmap(bmp) 
        }
    }

    private val pickDriving = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { 
            val bmp = uriToBitmap(it)
            drivingBitmap = bmp
            findViewById<ImageView>(R.id.imgDriving).setImageBitmap(bmp) 
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        engine = MotionEngine()
        initModels()

        findViewById<Button>(R.id.btnPickSource).setOnClickListener { pickSource.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
        findViewById<Button>(R.id.btnPickDriving).setOnClickListener { pickDriving.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
        
        findViewById<Button>(R.id.btnProcess).setOnClickListener {
            val src = sourceBitmap
            val drv = drivingBitmap
            if (src != null && drv != null) {
                val output = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
                engine.processFrame(src, drv, output)
                findViewById<ImageView>(R.id.imgOutput).setImageBitmap(output)
            }
        }
    }

    private fun uriToBitmap(uri: android.net.Uri): Bitmap {
        val inputStream = contentResolver.openInputStream(uri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        // CRITICAL: Resize to model requirement (256x256)
        return Bitmap.createScaledBitmap(bitmap, 256, 256, true)
    }

    private fun initModels() {
        val files = listOf("FOMMDetector.onnx", "FOMMGenerator.onnx")
        files.forEach { name ->
            val file = File(filesDir, name)
            if (!file.exists()) {
                assets.open(name).use { input -> file.outputStream().use { input.copyTo(it) } }
            }
        }
        engine.initEngine(File(filesDir, "FOMMDetector.onnx").absolutePath, File(filesDir, "FOMMGenerator.onnx").absolutePath)
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.releaseEngine()
    }
}
