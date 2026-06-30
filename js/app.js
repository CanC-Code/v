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

document.addEventListener("DOMContentLoaded", () => {
  console.log("MotionForge: App loaded. Configuring ORT...");
  
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
    console.log(`Status update [${kind || 'info'}]: ${msg}`);
    el.textContent = msg;
    el.className = "status" + (kind ? ` ${kind}` : "");
  }

  function updateRunButton() {
    const ready = isLoaded() && sourceReady && drivingReady;
    els.runBtn.disabled = !ready;
    console.log(`Run button state: ${ready ? 'Enabled' : 'Disabled'} (Loaded:${isLoaded()}, Src:${sourceReady}, Drv:${drivingReady})`);
  }

  // Initialize WebAssembly environment
  configureOrt();
  console.log("ORT Configured.");

  // ---------- Local Model Initialization ----------

  els.loadBtn.addEventListener("click", async () => {
    console.log("Initialize button clicked.");
    els.loadBtn.disabled = true;
    setStatus(els.modelStatus, "Loading models...", "pending");

    try {
      const providers = ["wasm"]; 
      console.log("Calling loadSessions...");

      // Ensure your file paths match these exactly
      await loadSessions('./models/FOMMDetector.onnx', './models/FOMMGenerator.onnx', providers);

      setStatus(els.modelStatus, `Initialized (${providers.join(", ")})`, "ok");
      console.log("Initialization sequence completed successfully.");
      updateRunButton();
    } catch (err) {
      console.error("CRITICAL Initialization error:", err);
      setStatus(els.modelStatus, `Load Error: ${err.message}`, "error");
    } finally {
      els.loadBtn.disabled = false;
    }
  });

  // ---------- Input Handling ----------

  els.sourceInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    console.log("Source image selected.");
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
    console.log("Driving video selected.");
    els.drivingVideo.src = URL.createObjectURL(file);
    els.drivingVideo.addEventListener("loadedmetadata", () => {
      drivingReady = true;
      updateRunButton();
    }, { once: true });
  });

  // ---------- Generation Pipeline ----------

  els.runBtn.addEventListener("click", async () => {
    console.log("Run button clicked.");
    els.runBtn.disabled = true;
    els.progressWrap.classList.remove("hidden");
    generatedFrames = [];

    console.time("TotalGeneration");
    try {
      const size = IO_CONFIG.frameSize;
      const sourceTensor = frameToTensor(els.sourceCanvas, size);
      console.log("Source tensor created.");

      // Pre-compute source keypoints
      const { kp: sourceKp, jac: sourceJac } = await computeSourceKeypoints(sourceTensor);
      console.log("Source keypoints calculated.");

      const video = els.drivingVideo;
      video.pause();

      const fps = 12;
      const frameCount = Math.max(1, Math.floor(video.duration * fps));
      const sampleCanvas = document.createElement("canvas");
      sampleCanvas.width = size;
      sampleCanvas.height = size;
      const sampleCtx = sampleCanvas.getContext("2d");

      console.log(`Starting generation for ${frameCount} frames.`);

      for (let i = 0; i < frameCount; i++) {
        video.currentTime = (i / frameCount) * video.duration;

        await new Promise((resolve) => {
            const timeout = setTimeout(() => {
                console.warn("Seek timed out, proceeding...");
                resolve();
            }, 500);
            video.onseeked = () => {
                clearTimeout(timeout);
                resolve();
            };
        });

        sampleCtx.drawImage(video, 0, 0, size, size);

        // Run inference
        const outTensor = await runFrame(sourceTensor, sourceKp, sourceJac, frameToTensor(sampleCanvas, size));

        const imageData = tensorToImageData(outTensor, size);
        generatedFrames.push(imageData);
        els.outputCanvas.getContext("2d").putImageData(imageData, 0, 0);

        const pct = Math.round(((i + 1) / frameCount) * 100);
        els.progressBar.value = pct;
        els.progressLabel.textContent = `${pct}%`;

        if (outTensor && typeof outTensor.dispose === 'function') outTensor.dispose();
      }

      els.exportBtn.disabled = false;
      setStatus(els.runStatus, "Generation complete.", "ok");
    } catch (err) {
      console.error("CRITICAL Generation error:", err);
      setStatus(els.runStatus, `Error: ${err.message}`, "error");
    } finally {
      els.runBtn.disabled = false;
      console.timeEnd("TotalGeneration");
    }
  });

  // ---------- Export Logic ----------
  // (Export logic remains unchanged)
  els.exportBtn.addEventListener("click", async () => {
      // ... (existing export code)
  });
});
