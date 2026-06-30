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
  console.log("MotionForge: App loaded.");

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
  }

  configureOrt();

  // Model Initialization
  els.loadBtn.addEventListener("click", async () => {
    els.loadBtn.disabled = true;
    setStatus(els.modelStatus, "Loading models...", "pending");

    try {
      const providers = ("gpu" in navigator) ? ["webgpu", "wasm"] : ["wasm"];
      await loadSessions('./models/FOMMDetector.onnx', './models/FOMMGenerator.onnx', providers);

      setStatus(els.modelStatus, `✅ Initialized (${providers.join(", ")})`, "ok");
      updateRunButton();
    } catch (err) {
      console.error("Init error:", err);
      setStatus(els.modelStatus, `Error: ${err.message}`, "error");
    } finally {
      els.loadBtn.disabled = false;
    }
  });

  // Input Handling
  els.sourceInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      els.sourceCanvas.getContext("2d").drawImage(img, 0, 0, 256, 256);
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

  // Generation
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
      const frameCount = Math.max(1, Math.floor(video.duration * fps));
      const sampleCanvas = document.createElement("canvas");
      sampleCanvas.width = size;
      sampleCanvas.height = size;
      const sampleCtx = sampleCanvas.getContext("2d");

      for (let i = 0; i < frameCount; i++) {
        video.currentTime = (i / frameCount) * video.duration;
        await new Promise(r => { video.onseeked = r; });

        sampleCtx.drawImage(video, 0, 0, size, size);

        const drivingTensor = frameToTensor(sampleCanvas, size);
        const outTensor = await runFrame(sourceTensor, sourceKp, sourceJac, drivingTensor);

        const imageData = tensorToImageData(outTensor, size);
        generatedFrames.push(imageData);
        els.outputCanvas.getContext("2d").putImageData(imageData, 0, 0);

        const pct = Math.round(((i + 1) / frameCount) * 100);
        els.progressBar.value = pct;
        els.progressLabel.textContent = `${pct}%`;

        if (outTensor?.dispose) outTensor.dispose();
        if (drivingTensor?.dispose) drivingTensor.dispose();
      }

      els.exportBtn.disabled = false;
      setStatus(els.runStatus, "Generation complete.", "ok");
    } catch (err) {
      console.error("Generation error:", err);
      setStatus(els.runStatus, `Error: ${err.message}`, "error");
    } finally {
      els.runBtn.disabled = false;
    }
  });

  // Export
  els.exportBtn.addEventListener("click", async () => {
    if (generatedFrames.length === 0) return;
    els.exportBtn.disabled = true;
    setStatus(els.runStatus, "Exporting WebM...", "pending");

    try {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = IO_CONFIG.frameSize;
      const ctx = canvas.getContext("2d");

      const stream = canvas.captureStream(12);
      const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
      const chunks = [];

      recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url; a.download = "motionforge.webm"; a.click();
        URL.revokeObjectURL(url);
        setStatus(els.runStatus, "✅ Exported", "ok");
      };

      recorder.start();
      for (const frame of generatedFrames) {
        ctx.putImageData(frame, 0, 0);
        await new Promise(r => setTimeout(r, 1000/12));
      }
      recorder.stop();
    } catch (err) {
      setStatus(els.runStatus, `Export failed: ${err.message}`, "error");
    } finally {
      els.exportBtn.disabled = false;
    }
  });
});