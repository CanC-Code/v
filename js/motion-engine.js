// motion-engine.js
// Finalized inference pipeline for Qualcomm FOMM ONNX assets.

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
    if (!response.ok) {
      throw new Error(`File not found: ${url} (HTTP ${response.status})`);
    }
    console.log(`✓ Verified: ${url}`);
    return true;
  } catch (err) {
    console.error(`✗ Verify failed for ${url}:`, err.message);
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

    if (onProgress) onProgress(80, "Creating Generator session...");

    genSession = await ort.InferenceSession.create(genModelUrl, { 
      executionProviders,
      externalData: [{ path: genDataPath, data: genDataUrl }]
    });

    if (onProgress) onProgress(100, "✅ Models ready");

    console.log("✅ Model initialization successful.");
    console.log("Generator Inputs:", genSession.inputNames);
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

// Map possible model-side input names
function buildFeeds(session, tensorMap) {
  const feeds = {};
  session.inputNames.forEach(name => {
    if (tensorMap[name]) {
      feeds[name] = tensorMap[name];
    } else {
      console.error(`ERROR: Model requires input '${name}'`);
    }
  });
  return feeds;
}

async function detectKeypoints(frameTensor) {
  const inputName = kpSession.inputNames[0];
  const results = await kpSession.run({ [inputName]: frameTensor });

  const kpKey = kpSession.outputNames.find(n => n.includes("keypoint") || n === "kp");
  const jacKey = kpSession.outputNames.find(n => n.includes("jacobian") || n === "jac");

  return { kp: results[kpKey], jac: jacKey ? results[jacKey] : null };
}

export async function runFrame(sourceTensor, sourceKp, sourceJac, drivingFrameTensor) {
  if (!isLoaded()) throw new Error("Models not loaded yet.");

  const { kp: drivingKp, jac: drivingJac } = await detectKeypoints(drivingFrameTensor);

  const tensorMapping = {
    "image": sourceTensor, "source_image": sourceTensor, "source": sourceTensor,
    "source_keypoints": sourceKp, "kp_source": sourceKp, "source_kp": sourceKp,
    "driving_keypoints": drivingKp, "kp_driving": drivingKp, "driving_kp": drivingKp,
    "source_jacobian": sourceJac, "driving_jacobian": drivingJac
  };

  const feeds = buildFeeds(genSession, tensorMapping);
  const results = await genSession.run(feeds);
  return results[genSession.outputNames[0]];
}

// Tensor Helpers
export function frameToTensor(canvas, size = 256) {
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;
  const tensorData = new Float32Array(size * size * 3);

  for (let i = 0; i < size * size; i++) {
    tensorData[i] = data[i*4] / 255;
    tensorData[i + size*size] = data[i*4 + 1] / 255;
    tensorData[i + size*size*2] = data[i*4 + 2] / 255;
  }
  return new ort.Tensor("float32", tensorData, [1, 3, size, size]);
}

export function tensorToImageData(tensor, size = 256) {
  const data = tensor.data;
  const imageData = new Uint8ClampedArray(size * size * 4);

  for (let i = 0; i < size * size; i++) {
    const idx = i * 4;
    imageData[idx] = Math.round(data[i] * 255);
    imageData[idx + 1] = Math.round(data[i + size*size] * 255);
    imageData[idx + 2] = Math.round(data[i + size*size*2] * 255);
    imageData[idx + 3] = 255;
  }
  return new ImageData(imageData, size, size);
}

export async function computeSourceKeypoints(sourceTensor) {
  return await detectKeypoints(sourceTensor);
}