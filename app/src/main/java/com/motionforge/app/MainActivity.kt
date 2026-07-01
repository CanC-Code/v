package com.example.motionforge

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
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
import com.motionforge.app.FommEngineWrapper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer

private const val FRAME_SIZE = 256
private const val OUTPUT_FRAME_RATE = 12
private const val MAX_FRAMES = 300 // ~25s at 12fps; keeps runaway durations from hanging the phone

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

            Divider(modifier = Modifier.padding(vertical = 8.dp))

            Button(
                onClick = {
                    if (sourceImageUri != null && drivingVideoUri != null) {
                        isGenerating = true
                        statusMessage = "Processing Native Motion Transference..."
                        resultVideoUri = null
                        coroutineScope.launch {
                            val result = executeMotionPipeline(context, sourceImageUri!!, drivingVideoUri!!) { msg ->
                                statusMessage = msg
                            }
                            if (result != null) {
                                resultVideoUri = result
                                statusMessage = "Generation Successful!"
                            } else {
                                statusMessage = "Pipeline Inference Failed. Check Logcat (tag: FommEngine)."
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

/**
 * Copies a model's .onnx graph AND its paired .data external-weights file (if present)
 * from assets into internal storage. ONNX Runtime resolves external data relative to
 * the .onnx file's directory using the filename baked into the graph at export time,
 * so both files must land side-by-side with matching names, or session construction
 * fails (often surfacing later as "Invalid fd" once inference tries to touch the
 * weights). baseName should be the filename WITHOUT extension, e.g. "FOMMDetector".
 *
 * IMPORTANT: scripts/fetch_models.sh copies files matched by `-name "*FOMMDetector.onnx"`,
 * which preserves whatever prefix existed inside the upstream zip. If the actual file in
 * app/src/main/assets/ isn't exactly "FOMMDetector.onnx" / "FOMMDetector.data", this will
 * throw FileNotFoundException on the .onnx open. Run `ls app/src/main/assets/` to confirm
 * the real filenames before assuming this function is the problem.
 */
fun getAssetPath(context: Context, baseName: String): String {
    val onnxFile = File(context.filesDir, "$baseName.onnx")
    if (!onnxFile.exists()) {
        context.assets.open("$baseName.onnx").use { input ->
            onnxFile.outputStream().use { input.copyTo(it) }
        }
    }

    val dataAssetName = "$baseName.data"
    val hasDataAsset = try {
        context.assets.open(dataAssetName).close()
        true
    } catch (e: java.io.FileNotFoundException) {
        false
    }

    if (hasDataAsset) {
        val dataFile = File(context.filesDir, dataAssetName)
        if (!dataFile.exists()) {
            context.assets.open(dataAssetName).use { input ->
                dataFile.outputStream().use { input.copyTo(it) }
            }
        }
    }

    return onnxFile.absolutePath
}

private fun bitmapToRgbaBytes(bitmap: Bitmap): ByteArray {
    val safeBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
        bitmap.copy(Bitmap.Config.ARGB_8888, false)
    } else {
        bitmap
    }
    val buffer = ByteBuffer.allocate(safeBitmap.byteCount)
    safeBitmap.copyPixelsToBuffer(buffer)
    return buffer.array()
}

/**
 * Decodes the source image and driving video, runs the native engine per-frame, and
 * encodes the results into an mp4. No such video-level pipeline previously existed in
 * either the Kotlin or C++ layers -- FommEngineWrapper only ever exposed single-frame
 * processFrame(), so this orchestration has to live here.
 */
private suspend fun executeMotionPipeline(
    context: Context,
    imgUri: Uri,
    vidUri: Uri,
    onStatus: (String) -> Unit
): Uri? = withContext(Dispatchers.IO) {
    var retriever: MediaMetadataRetriever? = null
    var encoder: VideoEncoder? = null
    try {
        android.util.Log.d("FommEngine", "Starting pipeline initialization...")
        val engine = FommEngineWrapper()

        val kpModelPath = getAssetPath(context, "FOMMDetector")
        val genModelPath = getAssetPath(context, "FOMMGenerator")

        if (!engine.initialize(kpModelPath, genModelPath)) {
            android.util.Log.e("FommEngine", "Failed to initialize ONNX sessions.")
            return@withContext null
        }

        // Source image
        val sourceOptions = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val decodedSource = context.contentResolver.openInputStream(imgUri)?.use {
            BitmapFactory.decodeStream(it, null, sourceOptions)
        } ?: return@withContext null
        val scaledSource = Bitmap.createScaledBitmap(decodedSource, FRAME_SIZE, FRAME_SIZE, true)
        val sourceBytes = bitmapToRgbaBytes(scaledSource)

        // Driving video needs a local file path for MediaMetadataRetriever
        val drivingFile = File(context.cacheDir, "input_driving.mp4")
        context.contentResolver.openInputStream(vidUri)?.use { input ->
            FileOutputStream(drivingFile).use { input.copyTo(it) }
        }

        retriever = MediaMetadataRetriever().apply { setDataSource(drivingFile.absolutePath) }
        val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            ?.toLongOrNull() ?: 0L
        if (durationMs <= 0L) {
            android.util.Log.e("FommEngine", "Could not read driving video duration.")
            return@withContext null
        }
        val frameCount = ((durationMs / 1000.0) * OUTPUT_FRAME_RATE).toInt().coerceIn(1, MAX_FRAMES)

        val outputFile = File(context.cacheDir, "output_generation.mp4")
        encoder = VideoEncoder(outputFile.absolutePath, FRAME_SIZE, FRAME_SIZE, OUTPUT_FRAME_RATE)

        val outputBytes = ByteArray(FRAME_SIZE * FRAME_SIZE * 4)
        var failedFrames = 0
        for (i in 0 until frameCount) {
            val timeUs = (i.toLong() * durationMs * 1000L) / frameCount
            val rawFrame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                ?: continue
            val scaledFrame = Bitmap.createScaledBitmap(rawFrame, FRAME_SIZE, FRAME_SIZE, true)
            val drivingBytes = bitmapToRgbaBytes(scaledFrame)

            val success = engine.processFrame(sourceBytes, drivingBytes, outputBytes, FRAME_SIZE, FRAME_SIZE)
            if (!success) {
                failedFrames++
                android.util.Log.e("FommEngine", "processFrame failed on frame $i")
            }
            encoder.encodeFrame(outputBytes)
            onStatus("Generating frame ${i + 1}/$frameCount...")
        }

        if (failedFrames == frameCount) {
            android.util.Log.e("FommEngine", "All frames failed native inference.")
            return@withContext null
        }

        if (outputFile.exists() && outputFile.length() > 0) Uri.fromFile(outputFile) else null
    } catch (e: Exception) {
        android.util.Log.e("FommEngine", "Exception during pipeline execution", e)
        null
    } finally {
        retriever?.release()
        try {
            encoder?.finish()
        } catch (e: Exception) {
            android.util.Log.e("FommEngine", "Error finalizing encoder", e)
        }
    }
}
