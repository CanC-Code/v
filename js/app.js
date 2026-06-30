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
    setStatus(els.modelStatus, "Loading models (approx 50MB)...", "pending");

    try {
      // FOMM is highly optimized for WASM. WebGPU support is experimental 
      // for these specific operations.
      const providers = ["wasm"]; 

      await loadSessions('./models/FOMMDetector.onnx', './models/FOMMGenerator.onnx', providers);

      setStatus(els.modelStatus, `Initialized (${providers.join(", ")})`, "ok");
      updateRunButton();
      console.log("Model session created successfully.");
    } catch (err) {
      console.error("Initialization error:", err);
      setStatus(els.modelStatus, `Load Error: ${err.message}`, "error");
    } finally {
      els.loadBtn.disabled = false;
    }
  });

  // ---------- Input Handling ----------

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

  // ---------- Generation Pipeline ----------

  els.runBtn.addEventListener("click", async () => {
    els.runBtn.disabled = true;
    els.progressWrap.classList.remove("hidden");
    generatedFrames = [];

    console.time("TotalGeneration");
    try {
      const size = IO_CONFIG.frameSize;
      const sourceTensor = frameToTensor(els.sourceCanvas, size);

      // Pre-compute source keypoints
      const { kp: sourceKp, jac: sourceJac } = await computeSourceKeypoints(sourceTensor);

      const video = els.drivingVideo;
      video.pause();

      const fps = 12;
      const frameCount = Math.max(1, Math.floor(video.duration * fps));
      const sampleCanvas = document.createElement("canvas");
      sampleCanvas.width = size;
      sampleCanvas.height = size;
      const sampleCtx = sampleCanvas.getContext("2d");

      for (let i = 0; i < frameCount; i++) {
        // Safe seeking logic with timeout
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

        // Update UI
        const pct = Math.round(((i + 1) / frameCount) * 100);
        els.progressBar.value = pct;
        els.progressLabel.textContent = `${pct}%`;

        // Manual cleanup if available
        if (outTensor && typeof outTensor.dispose === 'function') outTensor.dispose();
      }

      els.exportBtn.disabled = false;
      setStatus(els.runStatus, "Generation complete.", "ok");
    } catch (err) {
      console.error("Generation error:", err);
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

      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "motionforge-output.webm";
        a.click();
        URL.revokeObjectURL(url);
        setStatus(els.runStatus, "Exported.", "ok");
      };

      recorder.start();
      for (const frame of generatedFrames) {
        exportCtx.putImageData(frame, 0, 0);
        await new Promise((r) => setTimeout(r, 1000 / 12));
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
