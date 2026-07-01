package com.motionforge.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var engine: MotionEngine
    private var sourceBitmap: Bitmap? = null
    private var drivingVideoUri: Uri? = null

    // Dedicated Image Picker
    private val pickSource = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let {
            val bmp = uriToBitmap(it)
            if (bmp != null) {
                sourceBitmap = bmp
                findViewById<ImageView>(R.id.imgSource).setImageBitmap(sourceBitmap)
            } else {
                Toast.makeText(this, "Failed to decode image. Please select a valid photo.", Toast.LENGTH_LONG).show()
            }
        }
    }

    // Dedicated Video Picker
    private val pickDriving = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let {
            drivingVideoUri = it
            Toast.makeText(this, "Video Selected Successfully", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        engine = MotionEngine()
        initModelsInBackground()

        findViewById<Button>(R.id.btnPickSource).setOnClickListener { 
            pickSource.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) 
        }
        
        findViewById<Button>(R.id.btnPickDriving).setOnClickListener { 
            pickDriving.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) 
        }
        
        findViewById<Button>(R.id.btnProcess).setOnClickListener {
            if (sourceBitmap != null && drivingVideoUri != null) {
                Toast.makeText(this, "Processing Video... Please wait.", Toast.LENGTH_LONG).show()
                findViewById<Button>(R.id.btnProcess).isEnabled = false // Prevent multi-clicks
                processVideo(drivingVideoUri!!)
            } else {
                Toast.makeText(this, "Please select both source image and driving video.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun processVideo(uri: Uri) {
        CoroutineScope(Dispatchers.IO).launch {
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(this@MainActivity, uri)
                
                val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                val durationMs = durationStr?.toLong() ?: 0
                
                val outputFile = File(getExternalFilesDir(Environment.DIRECTORY_MOVIES), "MotionForge_${System.currentTimeMillis()}.mp4")
                
                val width = 256
                val height = 256
                val frameRate = 30
                val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
                format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                format.setInteger(MediaFormat.KEY_BIT_RATE, 2000000)
                format.setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
                format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

                val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
                codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                codec.start()

                val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
                var trackIndex = -1
                var muxerStarted = false

                val frameIntervalUs = 1000000L / frameRate
                val bufferInfo = MediaCodec.BufferInfo()
                var timeUs = 0L
                var inputEOS = false
                var outputEOS = false

                while (!outputEOS) {
                    if (!inputEOS) {
                        val inputBufferIndex = codec.dequeueInputBuffer(10000)
                        if (inputBufferIndex >= 0) {
                            val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                            
                            if (frame == null || timeUs > durationMs * 1000) {
                                codec.queueInputBuffer(inputBufferIndex, 0, 0, timeUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputEOS = true
                            } else {
                                val resizedFrame = Bitmap.createScaledBitmap(frame, 256, 256, true)
                                val outputBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
                                
                                engine.processFrame(sourceBitmap!!, resizedFrame, outputBitmap)
                                
                                withContext(Dispatchers.Main) {
                                    findViewById<ImageView>(R.id.imgOutput).setImageBitmap(outputBitmap)
                                }

                                val inputBuffer = codec.getInputBuffer(inputBufferIndex)
                                inputBuffer?.clear()
                                
                                val yuvData = getNV12(width, height, outputBitmap)
                                inputBuffer?.put(yuvData)
                                
                                codec.queueInputBuffer(inputBufferIndex, 0, yuvData.size, timeUs, 0)
                                timeUs += frameIntervalUs
                            }
                        }
                    }

                    var outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 10000)
                    while (outputBufferIndex >= 0 || outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        if (outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                            val newFormat = codec.outputFormat
                            trackIndex = muxer.addTrack(newFormat)
                            muxer.start()
                            muxerStarted = true
                        } else if (outputBufferIndex >= 0) {
                            val encodedData = codec.getOutputBuffer(outputBufferIndex)
                            if (encodedData != null && bufferInfo.size != 0) {
                                encodedData.position(bufferInfo.offset)
                                encodedData.limit(bufferInfo.offset + bufferInfo.size)
                                if (muxerStarted) {
                                    muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                                }
                            }
                            if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                outputEOS = true
                            }
                            codec.releaseOutputBuffer(outputBufferIndex, false)
                        }
                        
                        if (!outputEOS) {
                            outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 0)
                        } else {
                            break
                        }
                    }
                }

                codec.stop()
                codec.release()
                if (muxerStarted) {
                    muxer.stop()
                    muxer.release()
                }

                withContext(Dispatchers.Main) {
                    findViewById<Button>(R.id.btnProcess).isEnabled = true
                    Toast.makeText(this@MainActivity, "Saved to Movies: ${outputFile.name}", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    findViewById<Button>(R.id.btnProcess).isEnabled = true
                    Toast.makeText(this@MainActivity, "Error during processing: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                retriever.release()
            }
        }
    }

    private fun getNV12(inputWidth: Int, inputHeight: Int, bitmap: Bitmap): ByteArray {
        val argb = IntArray(inputWidth * inputHeight)
        bitmap.getPixels(argb, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        val yuv = ByteArray(inputWidth * inputHeight * 3 / 2)
        var yIndex = 0
        var uvIndex = inputWidth * inputHeight
        
        for (j in 0 until inputHeight) {
            for (i in 0 until inputWidth) {
                val pixel = argb[j * inputWidth + i]
                val r = (pixel shr 16) and 0xff
                val g = (pixel shr 8) and 0xff
                val b = pixel and 0xff
                
                val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                
                yuv[yIndex++] = y.coerceIn(0, 255).toByte()
                if (j % 2 == 0 && i % 2 == 0) {
                    yuv[uvIndex++] = u.coerceIn(0, 255).toByte()
                    yuv[uvIndex++] = v.coerceIn(0, 255).toByte()
                }
            }
        }
        return yuv
    }

    private fun uriToBitmap(uri: Uri): Bitmap? {
        return try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            if (bitmap != null) {
                Bitmap.createScaledBitmap(bitmap, 256, 256, true)
            } else {
                null
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun initModelsInBackground() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val files = listOf("FOMMDetector.onnx", "FOMMGenerator.onnx", "FOMMDetector.data", "FOMMGenerator.data")
                files.forEach { name ->
                    val file = File(filesDir, name)
                    if (!file.exists()) {
                        assets.open(name).use { input -> file.outputStream().use { output -> input.copyTo(output) } }
                    }
                }
                
                val kpPath = File(filesDir, "FOMMDetector.onnx").absolutePath
                val genPath = File(filesDir, "FOMMGenerator.onnx").absolutePath
                val isSuccess = engine.initEngine(kpPath, genPath)
                
                withContext(Dispatchers.Main) {
                    if (isSuccess) {
                        findViewById<Button>(R.id.btnProcess).isEnabled = true
                        Toast.makeText(this@MainActivity, "Engine Ready", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this@MainActivity, "Failed to initialize native engine.", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Error loading models: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.releaseEngine()
    }
}
