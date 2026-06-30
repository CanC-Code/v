// motion-engine.js
// Finalized inference pipeline for Qualcomm FOMM ONNX assets.

export const IO_CONFIG = {
  kpDetector: {
    inputName: "image",
    outputKeypoints: "keypoints",
    outputJacobian: "jacobian", 
  },
  generator: {
    inputSource: "image", 
    output: "image",
  },
  frameSize: 256,
};

let kpSession = null;
let genSession = null;

export function configureOrt() {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;
}

/**
 * Pre-flight check: Verify files exist before ORT tries to load them.
 */
async function verifyFile(url) {
    const response = await fetch(url, { method: 'HEAD' });
    if (!response.ok) throw new Error(`File not found: ${url} (${response.status})`);
    return true;
}

/**
 * Loads sessions via URL strings.
 */
export async function loadSessions(kpModelUrl, genModelUrl, executionProviders = ["wasm"]) {
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

  return { kpSession, genSession };
}

export function isLoaded() {
  return !!(kpSession && genSession);
}

// DYNAMIC INPUT MAPPING
// Maps runtime variables to model-specific graph keys
function buildFeeds(session, mapping) {
    const feeds = {};
    for (const expectedName of session.inputNames) {
        if (mapping[expectedName] !== undefined && mapping[expectedName] !== null) {
            feeds[expectedName] = mapping[expectedName];
        } else {
            console.warn(`Model expects input '${expectedName}' but no valid tensor was mapped.`);
        }
    }
    return feeds;
}

async function detectKeypoints(frameTensor) {
  const inputName = kpSession.inputNames[0]; // Usually 'image'
  const feeds = { [inputName]: frameTensor };
  const results = await kpSession.run(feeds);
  
  // Use the actual output name found in the model
  const kpKey = kpSession.outputNames[0]; 
  const jacKey = kpSession.outputNames.find(n => n.includes("jacobian"));

  const kp = results[kpKey];
  const jac = jacKey ? results[jacKey] : null;
  
  if (!kp) throw new Error(`Missing keypoint tensor. Available: ${Object.keys(results)}`);
  return { kp, jac };
}

export async function runFrame(sourceTensor, sourceKp, sourceJac, drivingFrameTensor) {
  if (!isLoaded()) throw new Error("Models not loaded yet.");
  
  const { kp: drivingKp, jac: drivingJac } = await detectKeypoints(drivingFrameTensor);
  
  // The generator expects specific naming conventions for the QAI Hub FOMM model
  const tensorMapping = {
      // Inputs
      "image": sourceTensor,
      "source_image": sourceTensor,
      
      // Keypoints (including the specific _values variations)
      "source_keypoints": sourceKp,
      "source_keypoint_values": sourceKp, 
      "driving_keypoints": drivingKp,
      "driving_keypoint_values": drivingKp,
      
      // Jacobians (only passed if they exist)
      "source_jacobian": sourceJac,
      "driving_jacobian": drivingJac
  };

  const feeds = buildFeeds(genSession, tensorMapping);
  const results = await genSession.run(feeds);
  
  const outKey = genSession.outputNames[0];
  const out = results[outKey];
  
  if (!out) throw new Error(`Missing output tensor. Available: ${Object.keys(results)}`);
  return out;
}
