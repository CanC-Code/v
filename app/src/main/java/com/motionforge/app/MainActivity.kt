package com.example.motionforge

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.VideoView
import android.media.MediaMetadataRetriever
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.provider.MediaStore
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
                        statusMessage = "Synthesizing Motion Transfer..."
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
                Text(if (isGenerating) "Processing Sequence..." else "Generate Animation")
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
    } catch (e: Exception) {
    }
    
    return onnxFile.absolutePath
}

private suspend fun executeMotionPipeline(context: Context, imgUri: Uri, vidUri: Uri): Uri? = withContext(Dispatchers.IO) {
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
        
        val outChannel = NIOUtils.writableFileChannel(outputFile.absolutePath)
        val encoder = AndroidSequenceEncoder(outChannel, Rational.R(15, 1))

        val retriever = MediaMetadataRetriever()
        context.contentResolver.openFileDescriptor(vidUri, "r")?.fileDescriptor?.let {
            retriever.setDataSource(it)
        }
        
        val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
        val durationMs = durationStr?.toLong() ?: 0L
        val fps = 15
        val frameIntervalUs = 1000000L / fps
        val totalFrames = ((durationMs * 1000L) / frameIntervalUs).toInt()
        
        val outputBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        
        for (i in 0 until totalFrames) {
            val timeUs = i * frameIntervalUs
            val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
            
            if (frame != null) {
                val scaledFrame = Bitmap.createScaledBitmap(frame, 256, 256, true)
                val swFrame = scaledFrame.copy(Bitmap.Config.ARGB_8888, false)
                
                val success = engine.processFrame(sourceSoftwareBitmap, swFrame, outputBitmap, i == 0)
                if (success) {
                    encoder.encodeImage(outputBitmap)
                }
            }
        }
        
        encoder.finish()
        NIOUtils.closeQuietly(outChannel)
        retriever.release()

        if (outputFile.exists()) Uri.fromFile(outputFile) else null

    } catch (e: Exception) {
        android.util.Log.e("FommEngine", "Exception during pipeline execution", e)
        null
    }
}
