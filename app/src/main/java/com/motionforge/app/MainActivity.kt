package com.motionforge.app

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.widget.*
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var engine: MotionEngine
    private var sourceBitmap: Bitmap? = null
    private var drivingVideoUri: Uri? = null

    // Hybrid Picker: Accepts Images and Videos
    private val pickVisual = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let {
            if (it.toString().contains("video")) {
                drivingVideoUri = it
                Toast.makeText(this, "Video Selected", Toast.LENGTH_SHORT).show()
            } else {
                sourceBitmap = uriToBitmap(it)
                findViewById<ImageView>(R.id.imgSource).setImageBitmap(sourceBitmap)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        engine = MotionEngine()
        initModels()

        findViewById<Button>(R.id.btnPickSource).setOnClickListener { 
            pickVisual.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) 
        }
        findViewById<Button>(R.id.btnPickDriving).setOnClickListener { 
            pickVisual.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) 
        }
        
        findViewById<Button>(R.id.btnProcess).setOnClickListener {
            if (sourceBitmap != null && drivingVideoUri != null) {
                processVideo(drivingVideoUri!!)
            }
        }
    }

    private fun processVideo(uri: Uri) {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(this, uri)
        
        // Loop through video duration (e.g., 30fps)
        val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0
        val frameInterval = 33333L // approx 30fps in microseconds
        
        for (timeUs in 0 until durationMs * 1000 step frameInterval) {
            val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            if (frame != null) {
                val resizedFrame = Bitmap.createScaledBitmap(frame, 256, 256, true)
                val output = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
                
                // Native Inference
                engine.processFrame(sourceBitmap!!, resizedFrame, output)
                
                // TODO: Feed 'output' into MediaCodec for encoding
            }
        }
        retriever.release()
    }

    private fun uriToBitmap(uri: Uri): Bitmap {
        val inputStream = contentResolver.openInputStream(uri)
        val bitmap = android.graphics.BitmapFactory.decodeStream(inputStream)
        return Bitmap.createScaledBitmap(bitmap, 256, 256, true)
    }

    private fun initModels() {
        // [Existing asset logic stays the same]
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
