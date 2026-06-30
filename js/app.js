// app.js
import { fetchModelBytes, clearAllModels } from "./model-cache.js";
import {
  configureOrt,
  loadSessions,
  isLoaded,
  frameToTensor,
  tensorToImageData,
  computeSourceKeypoints,
  runFrame,
  IO_CONFIG,
} from "./motion-engine.js";

const $ = (id) => document.getElementById(id);

const els = {
  kpUrl: $("kp-model-url"),
  genUrl: $("gen-model-url"),
  loadBtn: $("load-model-btn"),
  clearBtn: $("clear-cache-btn"),
  modelStatus: $("model-status"),

  sourceInput: $("source-image-input"),
  sourceCanvas: $("source-canvas"),
  drivingInput: $("driving-video-input"),
  drivingVideo: $("driving-video"),

  runBtn: $("run-btn"),
  progressWrap: $("progress-wrap"),
  progressBar: $("progress-bar"),
  progressLabel: $("progress-label"),
  runStatus: $("run-status"),

  outputCanvas: $("output-canvas"),
  exportBtn: $("export-btn"),
};

let sourceReady = false;
let drivingReady = false;
let generatedFrames = []; // ImageData[] collected during the last run

function setStatus(el, msg, kind) {
  el.textContent = msg;
  el.className = "status" + (kind ? ` ${kind}` : "");
}

function updateRunButton() {
  els.runBtn.disabled = !(isLoaded() && sourceReady && drivingReady);
}

configureOrt();

// ---------- Model loading ----------

els.loadBtn.addEventListener("click", async () => {
  const kpUrl = els.kpUrl.value.trim();
  const genUrl = els.genUrl.value.trim();
  if (!kpUrl || !genUrl) {
    setStatus(els.modelStatus, "Provide both model URLs first.", "error");
    return;
  }

  els.loadBtn.disabled = true;
  try {
    setStatus(els.modelStatus, "Downloading keypoint detector...");
    const kpBytes = await fetchModelBytes(kpUrl, (loaded, total) => {
      setStatus(els.modelStatus, `Keypoint detector: ${progressText(loaded, total)}`);
    });

    setStatus(els.modelStatus, "Downloading generator...");
    const genBytes = await fetchModelBytes(genUrl, (loaded, total) => {
      setStatus(els.modelStatus, `Generator: ${progressText(loaded, total)}`);
    });

    setStatus(els.modelStatus, "Initializing ONNX Runtime sessions (WASM)...");
    const providers = await pickExecutionProviders();
    await loadSessions(kpBytes, genBytes, providers);

    setStatus(els.modelStatus, `Models loaded and cached locally. Using providers: ${providers.join(", ")}.`, "ok");
  } catch (err) {
    console.error(err);
    setStatus(els.modelStatus, `Failed to load models: ${err.message}`, "error");
  } finally {
    els.loadBtn.disabled = false;
    updateRunButton();
  }
});

els.clearBtn.addEventListener("click", async () => {
  await clearAllModels();
  setStatus(els.modelStatus, "Cached models cleared. You'll need to re-download next time.", "ok");
});

async function pickExecutionProviders() {
  // Prefer WebGPU if present (much faster), fall back to WASM (always available, fully offline-capable).
  if ("gpu" in navigator) {
    try {
      return ["webgpu", "wasm"];
    } catch {
      return ["wasm"];
    }
  }
  return ["wasm"];
}

function progressText(loaded, total) {
  const mb = (n) => (n / (1024 * 1024)).toFixed(1);
  if (total) {
    const pct = Math.round((loaded / total) * 100);
    return `${mb(loaded)}MB / ${mb(total)}MB (${pct}%)`;
  }
  return `${mb(loaded)}MB`;
}

// ---------- Source image ----------

els.sourceInput.addEventListener("change", async () => {
  const file = els.sourceInput.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    const ctx = els.sourceCanvas.getContext("2d");
    ctx.clearRect(0, 0, els.sourceCanvas.width, els.sourceCanvas.height);
    ctx.drawImage(img, 0, 0, els.sourceCanvas.width, els.sourceCanvas.height);
    sourceReady = true;
    updateRunButton();
  };
  img.src = URL.createObjectURL(file);
});

// ---------- Driving video ----------

els.drivingInput.addEventListener("change", () => {
  const file = els.drivingInput.files[0];
  if (!file) return;
  els.drivingVideo.src = URL.createObjectURL(file);
  els.drivingVideo.addEventListener(
    "loadedmetadata",
    () => {
      drivingReady = true;
      updateRunButton();
    },
    { once: true }
  );
});

// ---------- Run pipeline ----------

els.runBtn.addEventListener("click", async () => {
  els.runBtn.disabled = true;
  els.progressWrap.classList.remove("hidden");
  els.exportBtn.disabled = true;
  generatedFrames = [];
  setStatus(els.runStatus, "Preparing...");

  try {
    const size = IO_CONFIG.frameSize;
    const sourceTensor = frameToTensor(els.sourceCanvas, size);
    const { kp: sourceKp, jac: sourceJac } = await computeSourceKeypoints(sourceTensor);

    const video = els.drivingVideo;
    video.pause();
    video.currentTime = 0;
    await waitForSeek(video);

    const fps = 12; // sampling rate for inference; raise for smoother output at the cost of time
    const duration = video.duration;
    const frameCount = Math.max(1, Math.floor(duration * fps));

    const outCtx = els.outputCanvas.getContext("2d");
    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = size;
    sampleCanvas.height = size;
    const sampleCtx = sampleCanvas.getContext("2d");

    for (let i = 0; i < frameCount; i++) {
      const t = (i / frameCount) * duration;
      video.currentTime = t;
      await waitForSeek(video);

      sampleCtx.drawImage(video, 0, 0, size, size);
      const drivingTensor = frameToTensor(sampleCanvas, size);

      const outTensor = await runFrame(sourceTensor, sourceKp, sourceJac, drivingTensor);
      const imageData = tensorToImageData(outTensor, size);
      generatedFrames.push(imageData);

      outCtx.putImageData(imageData, 0, 0);

      const pct = Math.round(((i + 1) / frameCount) * 100);
      els.progressBar.value = pct;
      els.progressLabel.textContent = `${pct}% (${i + 1}/${frameCount} frames)`;
      setStatus(els.runStatus, `Generating frame ${i + 1} of ${frameCount}...`);
    }

    setStatus(els.runStatus, `Done. Generated ${generatedFrames.length} frames.`, "ok");
    els.exportBtn.disabled = generatedFrames.length === 0;
  } catch (err) {
    console.error(err);
    setStatus(els.runStatus, `Error: ${err.message}`, "error");
  } finally {
    els.runBtn.disabled = false;
  }
});

function waitForSeek(video) {
  return new Promise((resolve) => {
    const handler = () => {
      video.removeEventListener("seeked", handler);
      resolve();
    };
    video.addEventListener("seeked", handler);
  });
}

// ---------- Export ----------

els.exportBtn.addEventListener("click", async () => {
  if (generatedFrames.length === 0) return;
  els.exportBtn.disabled = true;
  setStatus(els.runStatus, "Encoding WebM...");

  try {
    const size = IO_CONFIG.frameSize;
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = size;
    exportCanvas.height = size;
    const exportCtx = exportCanvas.getContext("2d");

    const stream = exportCanvas.captureStream(12);
    const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
    const chunks = [];
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);

    const stopped = new Promise((resolve) => (recorder.onstop = resolve));
    recorder.start();

    for (const frame of generatedFrames) {
      exportCtx.putImageData(frame, 0, 0);
      await new Promise((r) => setTimeout(r, 1000 / 12));
    }

    recorder.stop();
    await stopped;

    const blob = new Blob(chunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "motionforge-output.webm";
    a.click();
    URL.revokeObjectURL(url);

    setStatus(els.runStatus, "Exported.", "ok");
  } catch (err) {
    console.error(err);
    setStatus(els.runStatus, `Export failed: ${err.message}`, "error");
  } finally {
    els.exportBtn.disabled = false;
  }
});
