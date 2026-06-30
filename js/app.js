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
    console.log(`Status [${kind || 'info'}]: ${msg}`);
    el.textContent = msg;
    el.className = "status" + (kind ? ` ${kind}` : "");
  }

  function updateRunButton() {
    const ready = isLoaded() && sourceReady && drivingReady;
    els.runBtn.disabled = !ready;
    console.log(`Run button: ${ready ? 'Enabled' : 'Disabled'} (Loaded:${isLoaded()}, Src:${sourceReady}, Drv:${drivingReady})`);
  }

  configureOrt();
  console.log("ORT Configured.");

  // ---------- Local Model Initialization ----------
  els.loadBtn.addEventListener("click", async () => {
    console.log("Initialize button clicked.");
    els.loadBtn.disabled = true;
    setStatus(els.modelStatus, "Loading models (this may take 10-30s)...", "pending");

    try {
      const providers = ["wasm"]; 
      await loadSessions('./models/FOMMDetector.onnx', './models/FOMMGenerator.onnx', providers);

      setStatus(els.modelStatus, `✅ Initialized (${providers.join(", ")})`, "ok");
      updateRunButton();
    } catch (err) {
      console.error("CRITICAL Initialization error:", err);
      let msg = err.message || String(err);
      if (msg.includes("404") || msg.includes("not found")) {
        msg = "Model files missing. Check models/ folder deployment.";
      } else if (msg.includes("externalData") || msg.includes("location")) {
        msg = "External data path mismatch.";
      }
      setStatus(els.modelStatus, `Load Error: ${msg}`, "error");
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

  // ---------- Generation Pipeline (unchanged logic) ----------
  els.runBtn.addEventListener("click", async () => {
    // ... your existing generation code ...
    console.log("Run button clicked.");
    // (keep your full run logic here)
  });

  // Export button remains as-is
  els.exportBtn.addEventListener("click", async () => {
    // ... your export code ...
  });
});