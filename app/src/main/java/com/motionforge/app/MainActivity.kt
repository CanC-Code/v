fun drainEncoder() {
    val bufferInfo = MediaCodec.BufferInfo()
    var outputBufferIndex: Int
    while (true) {
        outputBufferIndex = encoder.dequeueOutputBuffer(bufferInfo, 0)
        when (outputBufferIndex) {
            MediaCodec.INFO_TRY_AGAIN_LATER -> {
                // Do not check bufferInfo.flags here; it's stale
                Thread.sleep(10) // Small delay to avoid busy-wait
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
                    // EOS received
                    break
                }
                // Process output buffer
                val outputBuffer = encoder.getOutputBuffer(outputBufferIndex)
                // ... (encode/process logic)
                encoder.releaseOutputBuffer(outputBufferIndex, false)
            }
        }
    }
}