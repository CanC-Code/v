package com.motionforge.app

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private lateinit var motionEngine: FommEngineWrapper
    private lateinit var outputView: ImageView // Added to actually see the result

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Setup a simple view so it's not actually blank
        outputView = ImageView(this)
        setContentView(outputView)

        motionEngine = FommEngineWrapper()

        // Use lifecycleScope to run work off the main thread
        lifecycleScope.launch(Dispatchers.Default) {
            val success = initializeEngine()
            if (success) {
                processTestFrame()
            }
        }
    }

    private suspend fun initializeEngine(): Boolean = withContext(Dispatchers.IO) {
        // ... (Keep the copyAssetsToFilesDir and initialization logic here) ...
        // Returns Boolean
    }

    private suspend fun processTestFrame() = withContext(Dispatchers.Default) {
        // ... (Keep the processTestFrame logic here) ...
        
        // After success, switch to main to update the UI
        withContext(Dispatchers.Main) {
            outputView.setImageBitmap(outputBitmap)
            Log.d("MainActivity", "Frame updated on UI")
        }
    }
}
