// motion-engine.js
export const IO_CONFIG = {
  frameSize: 256,
};

let kpSession = null;
let genSession = null;

export function configureOrt() {
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 8);
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'verbose';
}

async function verifyFile(url) {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    if (!response.ok) throw new Error(`File not found: ${url} (HTTP ${response.status})`);
    console.log(`✓ Verified: ${url}`);
    return true;
  } catch (err) {
    console.error(`✗ Verify failed: ${url}`, err.message);
    throw err;
  }
}

export async function loadSessions(kpModelUrl, genModelUrl, executionProviders = ["wasm"], onProgress = null) {
  try {
    if (onProgress) onProgress(10, "Verifying model files...");

    console.log("Starting model initialization...");
    await verifyFile(kpModelUrl);
    await verifyFile(genModelUrl);

    if (onProgress) onProgress(30, "Verifying external weights...");

    const kpDataUrl = kpModelUrl.replace('.onnx', '.data');
    const genDataUrl = genModelUrl.replace('.onnx', '.data');

    await verifyFile(kpDataUrl);
    await verifyFile(genDataUrl);

    if (onProgress) onProgress(50, "Creating KP Detector session...");

    const kpDataPath = kpModelUrl.split('/').pop().replace('.onnx', '.data');
    const genDataPath = genModelUrl.split('/').pop().replace('.onnx', '.data');

    kpSession = await ort.InferenceSession.create(kpModelUrl, { 
      executionProviders,
      externalData: [{ path: kpDataPath, data: kpDataUrl }]
    });

    if (onProgress) onProgress(75, "Creating Generator session...");

    genSession = await ort.InferenceSession.create(genModelUrl, { 
      executionProviders,
      externalData: [{ path: genDataPath, data: genDataUrl }]
    });

    if (onProgress) onProgress(100, "✅ Models ready");

    console.log("✅ Model initialization successful.");
    return { kpSession, genSession };
  } catch (err) {
    console.error("❌ Initialization Failed:", err);
    if (onProgress) onProgress(0, `❌ Error: ${err.message}`, true);
    throw err; 
  }
}

export function isLoaded() {
  return !!(kpSession && genSession);
}

// ... (rest of the file remains exactly the same as my previous response: buildFeeds, detectKeypoints, runFrame, frameToTensor, tensorToImageData, computeSourceKeypoints)