package com.example.motionforge

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer

/**
 * Minimal H.264 encoder over MediaCodec's ByteBuffer input path (not the Surface/EGL
 * path, to avoid pulling GL context management into a native/JNI-heavy project).
 *
 * CAVEAT: COLOR_FormatYUV420Flexible's actual byte layout (NV12 vs I420/YV12, stride
 * padding) is device/encoder dependent. This assumes tightly-packed NV12, which is the
 * common case but is NOT guaranteed on every device. If you see corrupted/green output
 * on a specific phone, query MediaCodecInfo.CodecCapabilities.colorFormats for the
 * chosen encoder and adjust the chroma interleave in argbToNv12 accordingly.
 */
class VideoEncoder(
    outputPath: String,
    private val width: Int,
    private val height: Int,
    private val frameRate: Int = 12,
    bitRate: Int = 2_000_000
) {
    private val codec: MediaCodec
    private val muxer: MediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    private var trackIndex = -1
    private var muxerStarted = false
    private var frameIndex = 0L
    private val bufferInfo = MediaCodec.BufferInfo()

    init {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
            setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
    }

    /** `argb` must be a tightly packed RGBA8888 buffer of size width*height*4. */
    fun encodeFrame(argb: ByteArray) {
        val nv12 = argbToNv12(argb, width, height)
        val inIndex = codec.dequeueInputBuffer(10_000)
        if (inIndex >= 0) {
            val inputBuffer = codec.getInputBuffer(inIndex)!!
            inputBuffer.clear()
            inputBuffer.put(nv12)
            val ptUs = frameIndex * 1_000_000L / frameRate
            codec.queueInputBuffer(inIndex, 0, nv12.size, ptUs, 0)
            frameIndex++
        }
        drainEncoder(false)
    }

    fun finish() {
        val inIndex = codec.dequeueInputBuffer(10_000)
        if (inIndex >= 0) {
            codec.queueInputBuffer(inIndex, 0, 0, frameIndex * 1_000_000L / frameRate,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
        }
        drainEncoder(true)
        codec.stop()
        codec.release()
        if (muxerStarted) muxer.stop()
        muxer.release()
    }

    private fun drainEncoder(endOfStream: Boolean) {
        while (true) {
            val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            when {
                outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream) return
                outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    trackIndex = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                    muxerStarted = true
                }
                outIndex >= 0 -> {
                    val encodedData = codec.getOutputBuffer(outIndex)!!
                    if (bufferInfo.size > 0 && muxerStarted) {
                        encodedData.position(bufferInfo.offset)
                        encodedData.limit(bufferInfo.offset + bufferInfo.size)
                        muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                    }
                    codec.releaseOutputBuffer(outIndex, false)
                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) return
                }
            }
        }
    }

    private fun argbToNv12(argb: ByteArray, w: Int, h: Int): ByteArray {
        val frameSize = w * h
        val out = ByteArray(frameSize + frameSize / 2)
        var yIndex = 0
        var uvIndex = frameSize
        var i = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val r = argb[i].toInt() and 0xFF
                val g = argb[i + 1].toInt() and 0xFF
                val b = argb[i + 2].toInt() and 0xFF
                i += 4
                val yVal = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                out[yIndex++] = yVal.coerceIn(0, 255).toByte()
                if (y % 2 == 0 && x % 2 == 0) {
                    val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                    val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                    out[uvIndex++] = u.coerceIn(0, 255).toByte()
                    out[uvIndex++] = v.coerceIn(0, 255).toByte()
                }
            }
        }
        return out
    }
}
