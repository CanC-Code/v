package com.example.motionforge

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
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
import org.jcodec.api.android.AndroidFrameGrab
import org.jcodec.api.android.AndroidSequenceEncoder
import org.jcodec.common.io.NIOUtils
import org.jcodec.common.model.Rational
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
    
    // UI State for Progression and Settings
    var isGenerating by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("") }
    var generationProgress by remember { mutableFloatStateOf(0f) }
    var etaSeconds by remember { mutableIntStateOf(0) }
    var maxDurationSeconds by remember { mutableIntStateOf(5) }

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
                                contentScale = ContentScale.Crop // Use ContentScale.Fit if cropping creates misaligned bounding boxes
                            )
                        } ?: Text("No Image", modifier = Modifier.align(Alignment.Center), color = Color.DarkGray)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(
                        onClick = { imagePicker.launch(androidx.activity.result.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !isGenerating
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
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !isGenerating
                    ) {
                        Text("Select Video")
                    }
                }
            }

            Divider(modifier = Modifier.padding(vertical = 8.dp))

            // Duration Setting
            Text("Output Duration: $maxDurationSeconds seconds", style = MaterialTheme.typography.titleMedium)
            Slider(
                value = maxDurationSeconds.toFloat(),
                onValueChange = { maxDurationSeconds = it.toInt() },
                valueRange = 1f..15f,
                steps = 13,
                modifier = Modifier.padding(horizontal = 8.dp),
                enabled = !isGenerating
            )

            Button(
                onClick = {
                    if (sourceImageUri != null && drivingVideoUri != null) {
                        isGenerating = true
                        generationProgress = 0f
                        etaSeconds = 0
                        statusMessage = "Initializing Pipeline..."
                        resultVideoUri = null
                        
                        coroutineScope.launch {
                            val result = executeMotionPipeline(
                                context = context, 
                                imgUri = sourceImageUri!!, 
                                vidUri = drivingVideoUri!!,
                                maxDuration = maxDurationSeconds,
                                onProgressUpdate = { progress, eta ->
                                    generationProgress = progress
                                    etaSeconds = eta
                                    statusMessage = "Synthesizing AI Frames... ${(progress * 100).toInt()}%"
                                }
                            )
                            
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
                Text(if (isGenerating) "Processing Sequence..." else "Generate Animation")
            }

            if (statusMessage.isNotEmpty()) {
                Text(text = statusMessage, style = MaterialTheme.typography.bodyMedium)
            }

            // Real-Time Progression Bar
            if (isGenerating) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = generationProgress,
                    modifier = Modifier
                        .fillMaxWidth(0.9f)
                        .height(8.dp)
                )
                Text(
                    text = "Estimated time remaining: ${etaSeconds}s",
                    style = MaterialTheme.typography.labelMedium,
                    modifier = Modifier.padding(top = 4.dp),
                    color = Color.Gray
                )
            }

            if (resultVideoUri != null && !isGenerating) {
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

fun getAssetPath(context: Context, baseName: String): String {
    val onnxFile = File(context.filesDir, "$baseName.onnx")
    if (!onnxFile.exists()) {
        context.assets.open("$baseName.onnx").use { input ->
            onnxFile.outputStream().use { input.copyTo(it) }
        }
    }

    val dataFile = File(context.filesDir, "$baseName.data")
    try {
        if (!dataFile.exists()) {
            context.assets.open("$baseName.data").use { input ->
                dataFile.outputStream().use { input.copyTo(it) }
            }
        }
    } catch (e: Exception) {}
    return onnxFile.absolutePath
}

private suspend fun executeMotionPipeline(
    context: Context, 
    imgUri: Uri, 
    vidUri: Uri,
    maxDuration: Int,
    onProgressUpdate: (Float, Int) -> Unit
): Uri? = withContext(Dispatchers.IO) {
    try {
        val engine = FommEngineWrapper()
        val kpModelPath = getAssetPath(context, "FOMMDetector")
        val genModelPath = getAssetPath(context, "FOMMGenerator")

        if (!engine.initialize(kpModelPath, genModelPath)) {
            android.util.Log.e("FommEngine", "Failed to initialize ONNX Sessions.")
            return@withContext null
        }

        var sourceBitmap: Bitmap?
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, imgUri)
            sourceBitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.setTargetSize(256, 256)
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } else {
            val bmp = MediaStore.Images.Media.getBitmap(context.contentResolver, imgUri)
            sourceBitmap = Bitmap.createScaledBitmap(bmp, 256, 256, true)
        }
        val sourceSoftwareBitmap = sourceBitmap!!.copy(Bitmap.Config.ARGB_8888, false)

        val outputFile = File(context.cacheDir, "output_generation.mp4")
        if (outputFile.exists()) outputFile.delete()
        
        // Cache driving video locally for JCodec parsing
        val tempDrivingFile = File(context.cacheDir, "temp_driving.mp4")
        context.contentResolver.openInputStream(vidUri)?.use { input ->
            FileOutputStream(tempDrivingFile).use { input.copyTo(it) }
        }

        val inChannel = NIOUtils.readableChannel(tempDrivingFile)
        val grabber = AndroidFrameGrab.createAndroidFrameGrab(inChannel)
        
        val outChannel = NIOUtils.writableFileChannel(outputFile.absolutePath)
        val encoder = AndroidSequenceEncoder(outChannel, Rational.R(15, 1))

        val fps = 15
        val targetFrames = maxDuration * fps
        var currentFrameCount = 0
        
        val outputBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        val startTime = System.currentTimeMillis()

        // Pure software decoding loop ensures perfectly extracted keyframes
        var frame = grabber.frame
        while (frame != null && currentFrameCount < targetFrames) {
            
            val scaledFrame = Bitmap.createScaledBitmap(frame, 256, 256, true)
            val swFrame = scaledFrame.copy(Bitmap.Config.ARGB_8888, false)
            
            val success = engine.processFrame(sourceSoftwareBitmap, swFrame, outputBitmap, currentFrameCount == 0)
            if (success) {
                encoder.encodeImage(outputBitmap)
            }
            
            currentFrameCount++
            
            // Re-eval metrics
            val elapsedMs = System.currentTimeMillis() - startTime
            val avgMsPerFrame = elapsedMs / currentFrameCount
            val remainingFrames = targetFrames - currentFrameCount
            val etaSec = (remainingFrames * avgMsPerFrame) / 1000
            val currentProgress = currentFrameCount.toFloat() / targetFrames.toFloat()
            
            withContext(Dispatchers.Main) {
                onProgressUpdate(currentProgress, etaSec.toInt())
            }
            
            frame = grabber.frame // Pull the next sequential frame from the video stream
        }
        
        encoder.finish()
        NIOUtils.closeQuietly(outChannel)
        NIOUtils.closeQuietly(inChannel)

        if (outputFile.exists()) Uri.fromFile(outputFile) else null

    } catch (e: Exception) {
        android.util.Log.e("FommEngine", "Exception during pipeline execution", e)
        null
    }
}
