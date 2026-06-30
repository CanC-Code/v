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

export async function loadSessions(kpModelUrl, genModelUrl, executionProviders = ["wasm"]) {
  try {
    console.log("Starting model initialization...");
    console.log("KP URL:", kpModelUrl);
    console.log("GEN URL:", genModelUrl);

    await verifyFile(kpModelUrl);
    await verifyFile(genModelUrl);

    const kpDataUrl = kpModelUrl.replace('.onnx', '.data');
    const genDataUrl = genModelUrl.replace('.onnx', '.data');

    await verifyFile(kpDataUrl);
    await verifyFile(genDataUrl);

    const kpDataPath = kpModelUrl.split('/').pop().replace('.onnx', '.data');
    const genDataPath = genModelUrl.split('/').pop().replace('.onnx', '.data');

    console.log("Using external data paths:", kpDataPath, genDataPath);

    kpSession = await ort.InferenceSession.create(kpModelUrl, { 
      executionProviders,
      externalData: [{ path: kpDataPath, data: kpDataUrl }]
    });

    genSession = await ort.InferenceSession.create(genModelUrl, { 
      executionProviders,
      externalData: [{ path: genDataPath, data: genDataUrl }]
    });

    console.log("✅ Model initialization successful.");
    console.log("Generator Inputs:", genSession.inputNames);
    console.log("Generator Outputs:", genSession.outputNames);
    
    return { kpSession, genSession };
  } catch (err) {
    console.error("❌ Initialization Failed:", err);
    throw err; 
  }
}

export function isLoaded() {
  return !!(kpSession && genSession);
}

// Map possible model-side input names to our internal variables
function buildFeeds(session, tensorMap) {
  const feeds = {};
  session.inputNames.forEach(name => {
    if (tensorMap[name]) {
      feeds[name] = tensorMap[name];
    } else {
      console.error(`ERROR: Model requires input '${name}', but it was not provided in the mapping.`);
    }
  });
  return feeds;
}

async function detectKeypoints(frameTensor) {
  const inputName = kpSession.inputNames[0]; 
  const results = await kpSession.run({ [inputName]: frameTensor });

  const kpKey = kpSession.outputNames.find(n => n.includes("keypoint") || n === "kp");
  const jacKey = kpSession.outputNames.find(n => n.includes("jacobian") || n === "jac");

  return { 
    kp: results[kpKey], 
    jac: jacKey ? results[jacKey] : null 
  };
}

export async function runFrame(sourceTensor, sourceKp, sourceJac, drivingFrameTensor) {
  if (!isLoaded()) throw new Error("Models not loaded yet.");

  const { kp: drivingKp, jac: drivingJac } = await detectKeypoints(drivingFrameTensor);

  const tensorMapping = {
    "image": sourceTensor,
    "source_image": sourceTensor,
    "source": sourceTensor,
    "input_image": sourceTensor,

    "source_keypoints": sourceKp,
    "source_keypoint_values": sourceKp,
    "kp_source": sourceKp,
    "source_kp": sourceKp,

    "driving_keypoints": drivingKp,
    "driving_keypoint_values": drivingKp,
    "kp_driving": drivingKp,
    "driving_kp": drivingKp,

    "source_jacobian": sourceJac,
    "jac_source": sourceJac,
    "jacobian_source": sourceJac,
    "driving_jacobian": drivingJac,
    "jac_driving": drivingJac,
    "jacobian_driving": drivingJac
  };

  const feeds = buildFeeds(genSession, tensorMapping);
  const results = await genSession.run(feeds);

  const outKey = genSession.outputNames[0];
  return results[outKey];
}

// === Tensor Utilities (CH4N) ===
export function frameToTensor(canvas, size = 256) {
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;

  // NHWC -> NCHW, normalized [0,1] as float32
  const tensorData = new Float32Array(size * size * 3);
  for (let i = 0; i < size * size; i++) {
    tensorData[i] = data[i * 4] / 255;           // R
    tensorData[i + size * size] = data[i * 4 + 1] / 255; // G
    tensorData[i + size * size * 2] = data[i * 4 + 2] / 255; // B
  }

  return new ort.Tensor("float32", tensorData, [1, 3, size, size]);
}

export function tensorToImageData(tensor, size = 256) {
  const data = tensor.data;
  const imageData = new Uint8ClampedArray(size * size * 4);

  for (let i = 0; i < size * size; i++) {
    const idx = i * 4;
    imageData[idx]     = Math.round(data[i] * 255);           // R
    imageData[idx + 1] = Math.round(data[i + size*size] * 255);     // G
    imageData[idx + 2] = Math.round(data[i + size*size*2] * 255);   // B
    imageData[idx + 3] = 255; // Alpha
  }

  return new ImageData(imageData, size, size);
}

export async function computeSourceKeypoints(sourceTensor) {
  if (!isLoaded()) throw new Error("Models not loaded");
  return await detectKeypoints(sourceTensor);
}