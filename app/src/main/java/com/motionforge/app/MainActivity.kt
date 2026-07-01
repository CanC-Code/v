package com.example.motionforge

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {

    // Load the native C++ library containing the ONNX runtime and FOMM logic
    init {
        try {
            System.loadLibrary("motionforge")
        } catch (e: UnsatisfiedLinkError) {
            e.printStackTrace()
        }
    }

    /**
     * JNI Hook: Triggers the native First Order Motion Model generation.
     * Ensure your native C++ file implements this exact signature.
     */
    private external fun generateMotionNative(
        sourceImagePath: String,
        drivingVideoPath: String,
        outputVideoPath: String
    ): Boolean

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MotionForgeScreen(::processMotionRequest)
                }
            }
        }
    }

    /**
     * Executes the heavy ONNX processing on a background thread.
     * This is critical to prevent OutOfMemory errors and UI thread blocking 
     * on mobile device architectures during tensor operations.
     */
    private suspend fun processMotionRequest(
        context: Context,
        imageUri: Uri,
        videoUri: Uri
    ): Uri? = withContext(Dispatchers.IO) {
        try {
            // 1. Copy content:// URIs to local cache so C++ can read standard file paths
            val imageFile = copyUriToCache(context, imageUri, "source_image.jpg")
            val videoFile = copyUriToCache(context, videoUri, "driving_video.mp4")
            val outputFile = File(context.cacheDir, "output_motion_${System.currentTimeMillis()}.mp4")

            if (imageFile == null || videoFile == null) return@withContext null

            // 2. Execute Native C++ Processing
            val success = generateMotionNative(
                imageFile.absolutePath,
                videoFile.absolutePath,
                outputFile.absolutePath
            )

            if (success && outputFile.exists()) {
                // 3. Clean up cache inputs to save storage
                imageFile.delete()
                videoFile.delete()
                return@withContext Uri.fromFile(outputFile)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return@withContext null
    }

    private fun copyUriToCache(context: Context, uri: Uri, fileName: String): File? {
        val file = File(context.cacheDir, fileName)
        return try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
            file
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MotionForgeScreen(
    onGenerate: suspend (Context, Uri, Uri) -> Uri?
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    var sourceImageUri by remember { mutableStateOf<Uri?>(null) }
    var drivingVideoUri by remember { mutableStateOf<Uri?>(null) }
    var outputVideoUri by remember { mutableStateOf<Uri?>(null) }
    var isProcessing by remember { mutableStateOf(false) }

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? -> sourceImageUri = uri }

    val videoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? -> drivingVideoUri = uri }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("MotionForge") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            
            // --- Input Section ---
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("1. Source Profile Image", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = { imagePicker.launch("image/*") }) {
                        Text(if (sourceImageUri != null) "Image Selected" else "Select Image")
                    }
                }
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text("2. Driving Movement Video", style = MaterialTheme.typography.titleMedium)
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = { videoPicker.launch("video/*") }) {
                        Text(if (drivingVideoUri != null) "Video Selected" else "Select Video")
                    }
                }
            }

            // --- Generation Section ---
            Button(
                onClick = {
                    if (sourceImageUri != null && drivingVideoUri != null) {
                        isProcessing = true
                        outputVideoUri = null
                        coroutineScope.launch {
                            val resultUri = onGenerate(context, sourceImageUri!!, drivingVideoUri!!)
                            outputVideoUri = resultUri
                            isProcessing = false
                            
                            if (resultUri == null) {
                                Toast.makeText(context, "Generation failed.", Toast.LENGTH_SHORT).show()
                            }
                        }
                    } else {
                        Toast.makeText(context, "Please select both image and video.", Toast.LENGTH_SHORT).show()
                    }
                },
                enabled = !isProcessing,
                modifier = Modifier.fillMaxWidth().height(56.dp)
            ) {
                if (isProcessing) {
                    CircularProgressIndicator(
                        color = MaterialTheme.colorScheme.onPrimary,
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text("Processing Tensor Models...")
                } else {
                    Text("Generate Motion")
                }
            }

            // --- Output Section ---
            if (outputVideoUri != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text("Generation Complete", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Button(
                            onClick = {
                                saveVideoToMediaStore(context, outputVideoUri!!)
                            }
                        ) {
                            Text("Save to Device Media")
                        }
                    }
                }
            }
        }
    }
}

/**
 * Saves the cached output video to the public Android MediaStore (Scoped Storage compliant).
 * This eliminates the need for legacy WRITE_EXTERNAL_STORAGE permissions on Android 10+.
 */
fun saveVideoToMediaStore(context: Context, cachedVideoUri: Uri) {
    val resolver = context.contentResolver
    val contentValues = ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, "MotionForge_${System.currentTimeMillis()}.mp4")
        put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
        put(MediaStore.MediaColumns.RELATIVE_PATH, "Movies/MotionForge")
    }

    val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)
    
    if (uri != null) {
        try {
            resolver.openOutputStream(uri)?.use { outputStream ->
                val cachedFile = File(cachedVideoUri.path!!)
                cachedFile.inputStream().use { inputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            Toast.makeText(context, "Saved to Movies/MotionForge", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(context, "Failed to save file.", Toast.LENGTH_SHORT).show()
        }
    }
}
