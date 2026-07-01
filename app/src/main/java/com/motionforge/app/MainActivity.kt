package com.example.motionforge

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.widget.VideoView
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import coil.compose.rememberAsyncImagePainter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MotionForgeApp()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MotionForgeApp() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    var sourceImageUri by remember { mutableStateOf<Uri?>(null) }
    var drivingVideoUri by remember { mutableStateOf<Uri?>(null) }
    var resultVideoUri by remember { mutableStateOf<Uri?>(null) }
    var isGenerating by remember { mutableStateOf(false) }

    // Launchers for picking media
    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) sourceImageUri = uri
    }
    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) drivingVideoUri = uri
    }

    // Launcher for saving the generated file securely using SAF
    val saveDocumentLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("video/mp4")) { destUri ->
        if (destUri != null && resultVideoUri != null) {
            coroutineScope.launch(Dispatchers.IO) {
                try {
                    context.contentResolver.openInputStream(resultVideoUri!!)?.use { input ->
                        context.contentResolver.openOutputStream(destUri)?.use { output ->
                            input.copyTo(output)
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("MotionForge") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(scrollState)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Input Selection Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Source Image Column
                Column(
                    modifier = Modifier.weight(1f),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(1f)
                            .background(Color.LightGray)
                    ) {
                        sourceImageUri?.let { uri ->
                            Image(
                                painter = rememberAsyncImagePainter(uri),
                                contentDescription = "Source Image",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Crop
                            )
                        } ?: Text(
                            "No Image",
                            modifier = Modifier.align(Alignment.Center),
                            color = Color.DarkGray
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(
                        onClick = {
                            imagePicker.launch(
                                androidx.activity.result.PickVisualMediaRequest(
                                    ActivityResultContracts.PickVisualMedia.ImageOnly
                                )
                            )
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Select Image")
                    }
                }

                // Driving Video Column
                Column(
                    modifier = Modifier.weight(1f),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(1f)
                            .background(Color.LightGray)
                    ) {
                        drivingVideoUri?.let { uri ->
                            VideoPreview(uri)
                        } ?: Text(
                            "No Video",
                            modifier = Modifier.align(Alignment.Center),
                            color = Color.DarkGray
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(
                        onClick = {
                            videoPicker.launch(
                                androidx.activity.result.PickVisualMediaRequest(
                                    ActivityResultContracts.PickVisualMedia.VideoOnly
                                )
                            )
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Select Video")
                    }
                }
            }

            Divider(modifier = Modifier.padding(vertical = 8.dp))

            // Generation Section
            Button(
                onClick = {
                    if (sourceImageUri != null && drivingVideoUri != null) {
                        isGenerating = true
                        resultVideoUri = null
                        coroutineScope.launch {
                            resultVideoUri = generateAnimation(context, sourceImageUri!!, drivingVideoUri!!)
                            isGenerating = false
                        }
                    }
                },
                enabled = sourceImageUri != null && drivingVideoUri != null && !isGenerating,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (isGenerating) "Processing Output..." else "Generate Animation")
            }

            if (isGenerating) {
                CircularProgressIndicator()
            }

            // Output Review & Save Section
            if (resultVideoUri != null) {
                Text("Result Preview:", style = MaterialTheme.typography.titleMedium)
                
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.8f)
                        .aspectRatio(1f)
                        .background(Color.Black)
                ) {
                    VideoPreview(resultVideoUri!!)
                }

                Button(
                    onClick = {
                        // Triggers the system UI to select a designated save location
                        saveDocumentLauncher.launch("MotionForge_Export_${System.currentTimeMillis()}.mp4")
                    },
                    modifier = Modifier.fillMaxWidth(0.8f)
                ) {
                    Text("Save Locally")
                }
            }
        }
    }
}

@Composable
fun VideoPreview(uri: Uri) {
    AndroidView(
        factory = { ctx ->
            VideoView(ctx).apply {
                setVideoURI(uri)
                setOnPreparedListener { mp ->
                    mp.isLooping = true
                    // Mute for preview purposes
                    mp.setVolume(0f, 0f)
                    start()
                }
            }
        },
        modifier = Modifier.fillMaxSize()
    )
}

/**
 * Placeholder processing pipeline.
 * Re-route your JNI ONNX integration here. For functional testing of the UI and save pipeline, 
 * this mimics the generation process by duplicating the input video to cache.
 */
suspend fun generateAnimation(context: Context, imageUri: Uri, videoUri: Uri): Uri = withContext(Dispatchers.IO) {
    // Simulate generation delay
    delay(2500)
    
    val tempFile = File(context.cacheDir, "temp_gen_${System.currentTimeMillis()}.mp4")
    
    context.contentResolver.openInputStream(videoUri)?.use { input ->
        FileOutputStream(tempFile).use { output ->
            input.copyTo(output)
        }
    }
    
    Uri.fromFile(tempFile)
}
