// MainActivity.kt
package com.motionforge.app

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setupEncoder()
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

    private fun drainEncoder() {
        if (!isEncoderStarted) return

        val bufferInfo = MediaCodec.BufferInfo()
        while (true) {
            val outputBufferIndex = encoder.dequeueOutputBuffer(bufferInfo, 0)
            when (outputBufferIndex) {
                MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    // Avoid busy-waiting
                    Thread.sleep(10)
                    continue
                }
                MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> {
                    // Handle buffer changes if needed
                    continue
                }
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    // Handle format changes if needed
                    continue
                }
                else -> {
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        Log.d("MainActivity", "EOS received, stopping encoder")
                        break
                    }
                    if (outputBufferIndex >= 0) {
                        val outputBuffer = encoder.getOutputBuffer(outputBufferIndex)
                        // Process output buffer (e.g., write to file or send to decoder)
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
    }
}