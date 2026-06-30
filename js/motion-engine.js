// motion-engine.js
export const IO_CONFIG = {
  frameSize: 256,
};

let kpSession = null;
let genSession = null;

export function configureOrt() {
  ort.env.wasm.numThreads = 4; // Reduced for faster startup
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'verbose';
}

async function verifyFile(url) {
  const response = await fetch(url, { method: 'HEAD' });
  if (!response.ok) throw new Error(`File not found: ${url} (${response.status})`);
  console.log(`✓ Verified: ${url}`);
}

export async function loadSessions(kpModelUrl, genModelUrl, executionProviders = ["wasm"], onProgress = null) {
  try {
    if (onProgress) onProgress(10, "Verifying files...");

    await verifyFile(kpModelUrl);
    await verifyFile(genModelUrl);

    const kpDataUrl = kpModelUrl.replace('.onnx', '.data');
    const genDataUrl = genModelUrl.replace('.onnx', '.data');

    await verifyFile(kpDataUrl);
    await verifyFile(genDataUrl);

    if (onProgress) onProgress(40, "Loading KP Detector (this may take 15-40s)...");

    const kpDataPath = kpModelUrl.split('/').pop().replace('.onnx', '.data');

    kpSession = await ort.InferenceSession.create(kpModelUrl, {
      executionProviders,
      externalData: [{ path: kpDataPath, data: kpDataUrl }],
      graphOptimizationLevel: 'all',
      executionMode: 'parallel'
    });

    if (onProgress) onProgress(75, "Loading Generator (this may take 20-50s)...");

    const genDataPath = genModelUrl.split('/').pop().replace('.onnx', '.data');

    genSession = await ort.InferenceSession.create(genModelUrl, {
      executionProviders,
      externalData: [{ path: genDataPath, data: genDataUrl }],
      graphOptimizationLevel: 'all',
      executionMode: 'parallel'
    });

    if (onProgress) onProgress(100, "✅ Models ready");
    console.log("✅ Initialization complete");
    return { kpSession, genSession };

  } catch (err) {
    console.error("❌ Load failed:", err);
    if (onProgress) onProgress(0, `❌ ${err.message}`, true);
    throw err;
  }
}

export function isLoaded() {
  return !!(kpSession && genSession);
}

// ... [Keep all the other functions exactly as in the previous full version: buildFeeds, detectKeypoints, runFrame, frameToTensor, tensorToImageData, computeSourceKeypoints]