package com.motionforge.app

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
    var statusMessage by remember { mutableStateOf("") }

    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) sourceImageUri = uri
    }
    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) drivingVideoUri = uri
    }

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
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
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
                        } ?: Text("No Image", modifier = Modifier.align(Alignment.Center), color = Color.DarkGray)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(
                        onClick = { imagePicker.launch(androidx.activity.result.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Select Image")
                    }
                }

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
                        drivingVideoUri?.let { uri -> VideoPreview(uri) }
                            ?: Text("No Video", modifier = Modifier.align(Alignment.Center), color = Color.DarkGray)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(
                        onClick = { videoPicker.launch(androidx.activity.result.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Select Video")
                    }
                }
            }

            // FIX: Reverted to Divider to match the 2023.08.00 Compose BOM Material 3 API
            Divider(modifier = Modifier.padding(vertical = 8.dp))

            Button(
                onClick = {
                    if (sourceImageUri != null && drivingVideoUri != null) {
                        isGenerating = true
                        statusMessage = "Processing Native Motion Transference..."
                        resultVideoUri = null
                        coroutineScope.launch {
                            val result = executeMotionPipeline(context, sourceImageUri!!, drivingVideoUri!!)
                            if (result != null) {
                                resultVideoUri = result
                                statusMessage = "Generation Successful!"
                            } else {
                                statusMessage = "Pipeline Inference Failed. Check Logcat."
                            }
                            isGenerating = false
                        }
                    }
                },
                enabled = sourceImageUri != null && drivingVideoUri != null && !isGenerating,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (isGenerating) "Processing Pipeline..." else "Generate Animation")
            }

            if (statusMessage.isNotEmpty()) {
                Text(text = statusMessage, style = MaterialTheme.typography.bodyMedium)
            }

            if (isGenerating) {
                CircularProgressIndicator()
            }

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
                    onClick = { saveDocumentLauncher.launch("MotionForge_Export_${System.currentTimeMillis()}.mp4") },
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
                    mp.setVolume(0f, 0f)
                    start()
                }
            }
        },
        modifier = Modifier.fillMaxSize()
    )
}

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

private suspend fun executeMotionPipeline(context: Context, imgUri: Uri, vidUri: Uri): Uri? = withContext(Dispatchers.IO) {
    try {
        android.util.Log.d("FommEngine", "Starting pipeline initialization...")
        val engine = FommEngineWrapper()
        
        val kpModelPath = getAssetPath(context, "FOMMDetector.onnx")
        val genModelPath = getAssetPath(context, "FOMMGenerator.onnx")

        val isInitialized = engine.initialize(kpModelPath, genModelPath)
        if (!isInitialized) {
            android.util.Log.e("FommEngine", "Failed to initialize ONNX Sessions.")
            return@withContext null
        }

        val sourceFile = File(context.cacheDir, "input_source.jpg")
        context.contentResolver.openInputStream(imgUri)?.use { input ->
            FileOutputStream(sourceFile).use { input.copyTo(it) }
        }

        val drivingFile = File(context.cacheDir, "input_driving.mp4")
        context.contentResolver.openInputStream(vidUri)?.use { input ->
            FileOutputStream(drivingFile).use { input.copyTo(it) }
        }

        val outputFile = File(context.cacheDir, "output_generation.mp4")

        val success = engine.processVideo(
            sourceImagePath = sourceFile.absolutePath,
            drivingVideoPath = drivingFile.absolutePath,
            outputPath = outputFile.absolutePath
        )

        if (success && outputFile.exists()) {
            Uri.fromFile(outputFile)
        } else {
            android.util.Log.e("FommEngine", "Native engine processVideo returned false.")
            null
        }
    } catch (e: Exception) {
        android.util.Log.e("FommEngine", "Exception during pipeline execution", e)
        null
    }
}
