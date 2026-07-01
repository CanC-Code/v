package com.example.motionforge

import android.content.Context
import java.io.File

object MotionForgeEngine {
    init {
        System.loadLibrary("motionforge")
    }

    /**
     * Native call to process motion transference.
     * @param kpModelPath Path to the keypoint detector model.
     * @param genModelPath Path to the dense motion generator model.
     * @param imagePath Path to the source image file.
     * @param videoPath Path to the driving video file.
     * @param outputPath Target path where the generated MP4 file should be written.
     * @return true if successful, false otherwise.
     */
    external fun runMotionTransfer(
        kpModelPath: String,
        genModelPath: String,
        imagePath: String,
        videoPath: String,
        outputPath: String
    ): Boolean

    // Utility helper to copy asset files to accessible internal storage paths
    fun getAssetPath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (!file.exists()) {
            context.assets.open(assetName).use { inputStream ->
                file.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return file.absolutePath
    }
}
