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

  function setStatus(el, msg, kind = "info") {
    console.log(`Status [${kind}]: ${msg}`);
    el.textContent = msg;
    el.className = "status" + (kind ? ` ${kind}` : "");
  }

  function updateRunButton() {
    const ready = isLoaded() && sourceReady && drivingReady;
    els.runBtn.disabled = !ready;
    console.log(`Run button state: ${ready ? 'Enabled' : 'Disabled'} (Loaded:${isLoaded()}, Src:${sourceReady}, Drv:${drivingReady})`);
  }

  // Initialize ORT
  configureOrt();
  console.log("ORT Configured.");

  // ---------- Local Model Initialization ----------
  els.loadBtn.addEventListener("click", async () => {
    console.log("Initialize button clicked.");
    els.loadBtn.disabled = true;
    setStatus(els.modelStatus, "Loading models (this may take 10-30s)...", "pending");

    try {
      // Prefer WebGPU when available, fallback to WASM
      const providers = ("gpu" in navigator) ? ["webgpu", "wasm"] : ["wasm"];
      console.log("Using providers:", providers);

      await loadSessions('./models/FOMMDetector.onnx', './models/FOMMGenerator.onnx', providers);

      setStatus(els.modelStatus, `✅ Initialized (${providers.join(", ")})`, "ok");
      console.log("Model sessions created successfully.");
      updateRunButton();
    } catch (err) {
      console.error("CRITICAL Initialization error:", err);
      let userMsg = err.message || String(err);
      
      if (userMsg.includes("404") || userMsg.includes("not found")) {
        userMsg = "Model files not found. Verify models/ folder was deployed.";
      } else if (userMsg.includes("externalData") || userMsg.includes("location")) {
        userMsg = "External data path mismatch. Check .data files.";
      } else if (userMsg.includes("WebGPU")) {
        userMsg = "WebGPU unavailable, falling back to WASM.";
      }
      
      setStatus(els.modelStatus, `Load Error: ${userMsg}`, "error");
    } finally {
      els.loadBtn.disabled = false;
    }
  });

  // ---------- Input Handling ----------
  els.sourceInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    console.log("Source image selected:", file.name);
    
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
    console.log("Driving video selected:", file.name);
    
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
            console.warn("Seek timed out for frame", i);
            resolve();
          }, 800);

          video.onseeked = () => {
            clearTimeout(timeout);
            resolve();
          };
        });

        sampleCtx.drawImage(video, 0, 0, size, size);

        // Run inference
        const drivingTensor = frameToTensor(sampleCanvas, size);
        const outTensor = await runFrame(sourceTensor, sourceKp, sourceJac, drivingTensor);

        const imageData = tensorToImageData(outTensor, size);
        generatedFrames.push(imageData);
        els.outputCanvas.getContext("2d").putImageData(imageData, 0, 0);

        const pct = Math.round(((i + 1) / frameCount) * 100);
        els.progressBar.value = pct;
        els.progressLabel.textContent = `${pct}%`;

        // Memory cleanup
        if (outTensor && typeof outTensor.dispose === 'function') outTensor.dispose();
        if (drivingTensor && typeof drivingTensor.dispose === 'function') drivingTensor.dispose();
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
  els.exportBtn.addEventListener("click", async () => {
    if (generatedFrames.length === 0) return;
    els.exportBtn.disabled = true;
    setStatus(els.runStatus, "Encoding WebM...", "pending");

    try {
      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = IO_CONFIG.frameSize;
      exportCanvas.height = IO_CONFIG.frameSize;
      const exportCtx = exportCanvas.getContext("2d");

      const stream = exportCanvas.captureStream(12);
      const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
      const chunks = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "motionforge-output.webm";
        a.click();
        URL.revokeObjectURL(url);
        setStatus(els.runStatus, "✅ Exported successfully.", "ok");
      };

      recorder.start();

      for (const frame of generatedFrames) {
        exportCtx.putImageData(frame, 0, 0);
        await new Promise(r => setTimeout(r, 1000 / 12));
      }

      recorder.stop();
    } catch (err) {
      console.error("Export error:", err);
      setStatus(els.runStatus, `Export failed: ${err.message}`, "error");
    } finally {
      els.exportBtn.disabled = false;
    }
  });
});