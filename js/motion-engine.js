// motion-engine.js
// Finalized inference pipeline for Qualcomm FOMM ONNX assets.

export const IO_CONFIG = {
  frameSize: 256,
};

let kpSession = null;
let genSession = null;

export function configureOrt() {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;
}

async function verifyFile(url) {
    const response = await fetch(url, { method: 'HEAD' });
    if (!response.ok) throw new Error(`File not found: ${url} (${response.status})`);
    return true;
}

/**
 * Loads sessions via URL strings.
 * Includes debug logging to surface initialization errors.
 */
export async function loadSessions(kpModelUrl, genModelUrl, executionProviders = ["wasm"]) {
  try {
    console.log("Starting model initialization...");
    await verifyFile(kpModelUrl);
    await verifyFile(genModelUrl);
    
    const kpDataFileName = kpModelUrl.split('/').pop().replace('.onnx', '.data');
    const genDataFileName = genModelUrl.split('/').pop().replace('.onnx', '.data');

    const kpDataUrl = kpModelUrl.replace('.onnx', '.data');
    const genDataUrl = genModelUrl.replace('.onnx', '.data');

    await verifyFile(kpDataUrl);
    await verifyFile(genDataUrl);

    kpSession = await ort.InferenceSession.create(kpModelUrl, { 
      executionProviders,
      externalData: [ { path: kpDataFileName, data: kpDataUrl } ]
    });
    
    genSession = await ort.InferenceSession.create(genModelUrl, { 
      executionProviders,
      externalData: [ { path: genDataFileName, data: genDataUrl } ]
    });

    console.log("Model initialization successful.");
    console.log("Generator Inputs expected:", genSession.inputNames);
    return { kpSession, genSession };
  } catch (err) {
    console.error("Initialization Failed:", err);
    throw err; // Re-throw so the UI can catch it
  }
}

export function isLoaded() {
  return !!(kpSession && genSession);
}

// Builds the feed object by matching available tensors to expected input names
function buildFeeds(session, tensorMap) {
    const feeds = {};
    session.inputNames.forEach(name => {
        if (tensorMap[name]) {
            feeds[name] = tensorMap[name];
        } else {
            console.error(`Missing required model input: '${name}'. Check tensor mapping.`);
        }
    });
    return feeds;
}

async function detectKeypoints(frameTensor) {
  const inputName = kpSession.inputNames[0]; 
  const results = await kpSession.run({ [inputName]: frameTensor });
  
  const kpKey = kpSession.outputNames.find(n => n.includes("keypoint"));
  const jacKey = kpSession.outputNames.find(n => n.includes("jacobian"));

  return { kp: results[kpKey], jac: jacKey ? results[jacKey] : null };
}

export async function runFrame(sourceTensor, sourceKp, sourceJac, drivingFrameTensor) {
  if (!isLoaded()) throw new Error("Models not loaded yet.");
  
  const { kp: drivingKp, jac: drivingJac } = await detectKeypoints(drivingFrameTensor);
  
  // This mapping is now exhaustive to ensure no missing inputs
  const tensorMapping = {
      "image": sourceTensor,
      "source_image": sourceTensor,
      "source_keypoints": sourceKp,
      "source_keypoint_values": sourceKp, 
      "driving_keypoints": drivingKp,
      "driving_keypoint_values": drivingKp,
      "source_jacobian": sourceJac,
      "driving_jacobian": drivingJac
  };

  const feeds = buildFeeds(genSession, tensorMapping);
  const results = await genSession.run(feeds);
  
  const outKey = genSession.outputNames[0];
  return results[outKey];
}
