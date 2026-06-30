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

// Ensure DOM is fully loaded before attaching listeners
document.addEventListener("DOMContentLoaded", () => {
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
      setStatus(els.modelStatus, "Loading models...", "pending");
      
      const [kpRes, genRes] = await Promise.all([
        fetch('./models/kp_detector.onnx'),
        fetch('./models/generator.onnx')
      ]);

      if (!kpRes.ok || !genRes.ok) throw new Error("Models not found. Check /models/ folder.");

      const [kpBytes, genBytes] = await Promise.all([
        kpRes.arrayBuffer(),
        genRes.arrayBuffer()
      ]);

      const providers = ("gpu" in navigator) ? ["webgpu", "wasm"] : ["wasm"];
      await loadSessions(kpBytes, genBytes, providers);

      setStatus(els.modelStatus, `Initialized via: ${providers.join(", ")}`, "ok");
      updateRunButton();
    } catch (err) {
      console.error(err);
      setStatus(els.modelStatus, `Load error: ${err.message}`, "error");
    } finally {
      els.loadBtn.disabled = false;
    }
  });

  // ---------- Inputs ----------

  els.sourceInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      const ctx = els.sourceCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0, 256, 256);
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

  // ---------- Generation ----------

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
      
      const fps = 12;
      const frameCount = Math.floor(video.duration * fps);
      const sampleCanvas = document.createElement("canvas");
      sampleCanvas.width = size;
      sampleCanvas.height = size;

      for (let i = 0; i < frameCount; i++) {
        video.currentTime = (i / frameCount) * video.duration;
        await new Promise(r => video.onseeked = r);

        sampleCanvas.getContext("2d").drawImage(video, 0, 0, size, size);
        const outTensor = await runFrame(sourceTensor, sourceKp, sourceJac, frameToTensor(sampleCanvas, size));
        
        const imageData = tensorToImageData(outTensor, size);
        generatedFrames.push(imageData);
        els.outputCanvas.getContext("2d").putImageData(imageData, 0, 0);

        const pct = Math.round(((i + 1) / frameCount) * 100);
        els.progressBar.value = pct;
        els.progressLabel.textContent = `${pct}%`;
      }
      els.exportBtn.disabled = false;
      setStatus(els.runStatus, "Done.", "ok");
    } catch (err) {
      setStatus(els.runStatus, err.message, "error");
    } finally {
      els.runBtn.disabled = false;
    }
  });
});
