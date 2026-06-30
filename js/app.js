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
  loadBtn: $("load-model-btn"),
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
let generatedFrames = [];

function setStatus(el, msg, kind) {
  el.textContent = msg;
  el.className = "status" + (kind ? ` ${kind}` : "");
}

function updateRunButton() {
  els.runBtn.disabled = !(isLoaded() && sourceReady && drivingReady);
}

configureOrt();

// ---------- Local Model Initialization ----------

els.loadBtn.addEventListener("click", async () => {
  els.loadBtn.disabled = true;
  try {
    setStatus(els.modelStatus, "Loading models from local storage...", "pending");
    
    // Fetch directly from the pre-deployed models folder
    const [kpResponse, genResponse] = await Promise.all([
      fetch('./models/kp_detector.onnx'),
      fetch('./models/generator.onnx')
    ]);

    if (!kpResponse.ok || !genResponse.ok) throw new Error("Could not find local model files.");

    const kpBytes = await kpResponse.arrayBuffer();
    const genBytes = await genResponse.arrayBuffer();

    const providers = await pickExecutionProviders();
    await loadSessions(kpBytes, genBytes, providers);

    setStatus(els.modelStatus, `Models initialized. Providers: ${providers.join(", ")}.`, "ok");
    updateRunButton();
  } catch (err) {
    console.error(err);
    setStatus(els.modelStatus, `Failed to load models: ${err.message}`, "error");
  } finally {
    els.loadBtn.disabled = false;
  }
});

async function pickExecutionProviders() {
  return ("gpu" in navigator) ? ["webgpu", "wasm"] : ["wasm"];
}

// ---------- Input Handling (Source Image / Driving Video) ----------

els.sourceInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    const ctx = els.sourceCanvas.getContext("2d");
    ctx.drawImage(img, 0, 0, els.sourceCanvas.width, els.sourceCanvas.height);
    sourceReady = true;
    updateRunButton();
  };
  img.src = URL.createObjectURL(file);
});

els.drivingInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  els.drivingVideo.src = URL.createObjectURL(file);
  els.drivingVideo.addEventListener("loadedmetadata", () => {
    drivingReady = true;
    updateRunButton();
  }, { once: true });
});

// ---------- Generation Pipeline ----------

els.runBtn.addEventListener("click", async () => {
  els.runBtn.disabled = true;
  els.progressWrap.classList.remove("hidden");
  generatedFrames = [];
  
  try {
    const size = IO_CONFIG.frameSize;
    const sourceTensor = frameToTensor(els.sourceCanvas, size);
    const { kp: sourceKp, jac: sourceJac } = await computeSourceKeypoints(sourceTensor);

    const video = els.drivingVideo;
    video.pause();
    video.currentTime = 0;
    
    const fps = 12;
    const frameCount = Math.floor(video.duration * fps);
    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = size;
    sampleCanvas.height = size;

    for (let i = 0; i < frameCount; i++) {
      video.currentTime = (i / frameCount) * video.duration;
      await new Promise(r => video.onseeked = r);

      sampleCanvas.getContext("2d").drawImage(video, 0, 0, size, size);
      const drivingTensor = frameToTensor(sampleCanvas, size);
      
      const outTensor = await runFrame(sourceTensor, sourceKp, sourceJac, drivingTensor);
      generatedFrames.push(tensorToImageData(outTensor, size));
      els.outputCanvas.getContext("2d").putImageData(generatedFrames[i], 0, 0);

      const pct = Math.round(((i + 1) / frameCount) * 100);
      els.progressBar.value = pct;
      els.progressLabel.textContent = `${pct}%`;
    }
    els.exportBtn.disabled = false;
    setStatus(els.runStatus, "Generation complete.", "ok");
  } catch (err) {
    setStatus(els.runStatus, err.message, "error");
  } finally {
    els.runBtn.disabled = false;
  }
});
