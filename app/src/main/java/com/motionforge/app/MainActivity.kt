// MainActivity.kt
package com.motionforge.app

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var encoder: MediaCodec
    private var isEncoderStarted = false
    private var fommEngine: FommEngine? = null

    // Load native library
    init {
        System.loadLibrary("motionforge_engine")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupEncoder()
        initializeFommEngine()
    }

    private fun setupEncoder() {
        try {
            val mimeType = MediaFormat.MIMETYPE_VIDEO_AVC
            val format = MediaFormat.createVideoFormat(mimeType, 1280, 720)
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            format.setInteger(MediaFormat.KEY_BIT_RATE, 1_000_000)
            format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

            encoder = MediaCodec.createEncoderByType(mimeType)
            encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            encoder.start()
            isEncoderStarted = true
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to setup encoder: ${e.message}")
        }
    }

    private fun initializeFommEngine() {
        try {
            // Replace with actual paths to your ONNX models
            val kpModelPath = "$filesDir/kp_model.onnx"
            val genModelPath = "$filesDir/gen_model.onnx"
            fommEngine = FommEngine()
            if (!fommEngine!!.initialize(kpModelPath, genModelPath)) {
                Log.e("MainActivity", "Failed to initialize FommEngine")
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Exception in initializeFommEngine: ${e.message}")
        }
    }

    private fun processFrame(sourceBitmap: Bitmap, drivingBitmap: Bitmap): Bitmap? {
        if (fommEngine == null) {
            Log.e("MainActivity", "FommEngine not initialized")
            return null
        }

        val width = sourceBitmap.width
        val height = sourceBitmap.height
        val outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        try {
            val result = fommEngine!!.processFrame(
                sourceBitmap.pixels, // Assuming pixels are accessible (simplified)
                drivingBitmap.pixels,
                outputBitmap.pixels,
                width,
                height
            )
            if (!result) {
                Log.e("MainActivity", "Failed to process frame")
                return null
            }
            return outputBitmap
        } catch (e: Exception) {
            Log.e("MainActivity", "Exception in processFrame: ${e.message}")
            return null
        }
    }

    private fun drainEncoder() {
        if (!isEncoderStarted) return

        val bufferInfo = MediaCodec.BufferInfo()
        while (true) {
            val outputBufferIndex = encoder.dequeueOutputBuffer(bufferInfo, 0)
            when (outputBufferIndex) {
                MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    Thread.sleep(10)
                    continue
                }
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    continue
                }
                else -> {
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        Log.d("MainActivity", "EOS received, stopping encoder")
                        break
                    }
                    if (outputBufferIndex >= 0) {
                        encoder.releaseOutputBuffer(outputBufferIndex, false)
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isEncoderStarted) {
            encoder.stop()
            encoder.release()
            isEncoderStarted = false
        }
        fommEngine?.let {
            // No explicit cleanup needed for FommEngine (handled in destructor)
        }
    }
}