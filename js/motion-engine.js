// motion-engine.js
export const IO_CONFIG = {
  frameSize: 256,
};

let kpSession = null;
let genSession = null;

export function configureOrt() {
  ort.env.wasm.numThreads = 4;
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'warning';
}

async function verifyFile(url) {
  const response = await fetch(url, { method: 'HEAD' });
  if (!response.ok) throw new Error(`File not found: ${url}`);
  console.log(`✓ Verified: ${url}`);
}

export async function loadSessions(kpModelUrl, genModelUrl, executionProviders = ["wasm"], onProgress = null) {
  try {
    if (onProgress) onProgress(10, "Verifying files...");
    await verifyFile(kpModelUrl);
    await verifyFile(genModelUrl);

    if (onProgress) onProgress(25, "Verifying weights...");

    const kpDataUrl = kpModelUrl.replace('.onnx', '.data');
    const genDataUrl = genModelUrl.replace('.onnx', '.data');

    await verifyFile(kpDataUrl);
    await verifyFile(genDataUrl);

    if (onProgress) onProgress(40, "Creating KP Detector session...");

    const kpDataPath = kpModelUrl.split('/').pop().replace('.onnx', '.data');
    const genDataPath = genModelUrl.split('/').pop().replace('.onnx', '.data');

    kpSession = await ort.InferenceSession.create(kpModelUrl, {
      executionProviders,
      externalData: [{ path: kpDataPath, data: kpDataUrl }],
      graphOptimizationLevel: 'all'
    });

    if (onProgress) onProgress(75, "Creating Generator session...");

    genSession = await ort.InferenceSession.create(genModelUrl, {
      executionProviders,
      externalData: [{ path: genDataPath, data: genDataUrl }],
      graphOptimizationLevel: 'all'
    });

    if (onProgress) onProgress(100, "✅ Models ready");
    console.log("✅ Initialization successful");
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

function buildFeeds(session, tensorMap) {
  const feeds = {};
  session.inputNames.forEach(name => {
    if (tensorMap[name]) {
      feeds[name] = tensorMap[name];
    } else {
      console.error(`Missing input: ${name}`);
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
  if (!isLoaded()) throw new Error("Models not loaded.");

  const { kp: drivingKp, jac: drivingJac } = await detectKeypoints(drivingFrameTensor);

  // Comprehensive mapping for Qualcomm FOMM model
  const tensorMapping = {
    "image": sourceTensor,
    "source_image": sourceTensor,
    "source": sourceTensor,

    "source_keypoints": sourceKp,
    "source_keypoint_values": sourceKp,
    "kp_source": sourceKp,
    "source_kp": sourceKp,

    "driving_keypoints": drivingKp,
    "driving_keypoint_values": drivingKp,
    "kp_driving": drivingKp,
    "driving_kp": drivingKp,

    // Jacobian names
    "source_jacobian": sourceJac,
    "source_keypoint_jacobians": sourceJac,   // ← Fixed
    "jac_source": sourceJac,
    "jacobian_source": sourceJac,

    "driving_jacobian": drivingJac,
    "driving_keypoint_jacobians": drivingJac,
    "jac_driving": drivingJac,
    "jacobian_driving": drivingJac
  };

  const feeds = buildFeeds(genSession, tensorMapping);
  const results = await genSession.run(feeds);
  return results[genSession.outputNames[0]];
}

// Tensor Utils
export function frameToTensor(canvas, size = 256) {
  const ctx = canvas.getContext("2d");
  const imgData = ctx.getImageData(0, 0, size, size);
  const data = imgData.data;
  const tensorData = new Float32Array(size * size * 3);
  for (let i = 0; i < size * size; i++) {
    const j = i * 4;
    tensorData[i] = data[j] / 255;
    tensorData[i + size*size] = data[j+1] / 255;
    tensorData[i + size*size*2] = data[j+2] / 255;
  }
  return new ort.Tensor("float32", tensorData, [1, 3, size, size]);
}

export function tensorToImageData(tensor, size = 256) {
  const data = tensor.data;
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const idx = i * 4;
    out[idx] = Math.round(data[i] * 255);
    out[idx+1] = Math.round(data[i + size*size] * 255);
    out[idx+2] = Math.round(data[i + size*size*2] * 255);
    out[idx+3] = 255;
  }
  return new ImageData(out, size, size);
}

export async function computeSourceKeypoints(sourceTensor) {
  return await detectKeypoints(sourceTensor);
}