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
    el.textContent = msg;
    el.className = "status" + (kind ? ` ${kind}` : "");
  }

  function updateRunButton() {
    els.runBtn.disabled = !(isLoaded() && sourceReady && drivingReady);
  }

  configureOrt();

  els.loadBtn.addEventListener("click", async () => {
    els.loadBtn.disabled = true;
    els.progressWrap.classList.remove("hidden");
    const updateProgress = (p, msg, err = false) => {
      els.progressBar.value = p;
      els.progressLabel.textContent = `${Math.round(p)}%`;
      setStatus(els.modelStatus, msg, err ? "error" : "pending");
    };

    try {
      const providers = ["wasm"];
      await loadSessions('./models/FOMMDetector.onnx', './models/FOMMGenerator.onnx', providers, updateProgress);
      setStatus(els.modelStatus, `✅ Initialized`, "ok");
      updateRunButton();
    } catch (e) {
      setStatus(els.modelStatus, `Failed: ${e.message}`, "error");
    } finally {
      els.loadBtn.disabled = false;
      setTimeout(() => els.progressWrap.classList.add("hidden"), 1500);
    }
  });

  els.sourceInput.addEventListener("change", e => {
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

  els.drivingInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (!file) return;
    els.drivingVideo.src = URL.createObjectURL(file);
    els.drivingVideo.addEventListener("loadedmetadata", () => {
      drivingReady = true;
      updateRunButton();
    }, { once: true });
  });

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
      const frameCount = Math.max(1, Math.floor(video.duration * 12));
      const sampleCanvas = document.createElement("canvas");
      sampleCanvas.width = sampleCanvas.height = size;
      const sampleCtx = sampleCanvas.getContext("2d");

      setStatus(els.runStatus, `Generating ${frameCount} frames...`, "pending");

      for (let i = 0; i < frameCount; i++) {
        video.currentTime = (i / frameCount) * video.duration;
        await new Promise(r => video.onseeked = r);

        sampleCtx.drawImage(video, 0, 0, size, size);

        const drivingTensor = frameToTensor(sampleCanvas, size);
        const outTensor = await runFrame(sourceTensor, sourceKp, sourceJac, drivingTensor);

        const imageData = tensorToImageData(outTensor, size);
        generatedFrames.push(imageData);
        els.outputCanvas.getContext("2d").putImageData(imageData, 0, 0);

        const pct = Math.round(((i + 1) / frameCount) * 100);
        els.progressBar.value = pct;
        els.progressLabel.textContent = `${pct}%`;
        setStatus(els.runStatus, `Generating frame ${i+1}/${frameCount}...`, "pending");
      }

      els.exportBtn.disabled = false;
      setStatus(els.runStatus, `✅ Done! ${generatedFrames.length} frames generated.`, "ok");
    } catch (err) {
      console.error(err);
      setStatus(els.runStatus, `Generation failed: ${err.message}`, "error");
    } finally {
      els.runBtn.disabled = false;
      setTimeout(() => els.progressWrap.classList.add("hidden"), 1500);
    }
  });

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

      recorder.ondataavailable = e => {
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
        setStatus(els.runStatus, "✅ Export successful!", "ok");
      };

      recorder.start();

      for (const frame of generatedFrames) {
        ctx.putImageData(frame, 0, 0);
        await new Promise(r => setTimeout(r, 60));
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