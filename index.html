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

    // Must match the "location" stored inside the .onnx protobuf
    const kpDataPath = kpModelUrl.split('/').pop().replace('.onnx', '.data');
    const genDataPath = genModelUrl.split('/').pop().replace('.onnx', '.data');

    console.log("Using external data paths:", kpDataPath, genDataPath);

    kpSession = await ort.InferenceSession.create(kpModelUrl, { 
      executionProviders,
      externalData: [{ 
        path: kpDataPath, 
        data: kpDataUrl 
      }]
    });

    genSession = await ort.InferenceSession.create(genModelUrl, { 
      executionProviders,
      externalData: [{ 
        path: genDataPath, 
        data: genDataUrl 
      }]
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

// Placeholder - implement these or keep your existing implementations
export function frameToTensor(canvas, size) {
  // TODO: your canvas → tensor conversion
  console.warn("frameToTensor not fully implemented in this snippet");
  return null;
}

export function tensorToImageData(tensor, size) {
  // TODO: your tensor → ImageData conversion
  console.warn("tensorToImageData not fully implemented in this snippet");
  return new ImageData(new Uint8ClampedArray(size * size * 4), size, size);
}

export async function computeSourceKeypoints(sourceTensor) {
  // TODO: implement if needed
  console.warn("computeSourceKeypoints placeholder");
  return { kp: null, jac: null };
}