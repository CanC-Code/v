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

export async function loadSessions(kpModelUrl, genModelUrl, executionProviders = ["wasm"]) {
  try {
    console.log("Starting model initialization...");
    await verifyFile(kpModelUrl);
    await verifyFile(genModelUrl);
    
    const kpDataFileName = kpModelUrl.split('/').pop().replace('.onnx', '.data');
    const genDataFileName = genModelUrl.split('/').pop().replace('.onnx', '.data');

    const kpDataUrl = kpModelUrl.replace('.onnx', '.data');
    const genDataUrl = genModelUrl.replace('.onnx', '.data');

    // Ensure weights exist
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
    console.log("Generator Inputs expected by ONNX model:", genSession.inputNames);
    return { kpSession, genSession };
  } catch (err) {
    console.error("Initialization Failed:", err);
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
        // Match the model's required input name against our broad alias map
        if (tensorMap[name]) {
            feeds[name] = tensorMap[name];
        } else {
            // This error is key: it tells you exactly what name to add to tensorMapping below
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

  return { kp: results[kpKey], jac: jacKey ? results[jacKey] : null };
}

export async function runFrame(sourceTensor, sourceKp, sourceJac, drivingFrameTensor) {
  if (!isLoaded()) throw new Error("Models not loaded yet.");
  
  const { kp: drivingKp, jac: drivingJac } = await detectKeypoints(drivingFrameTensor);
  
  // BROAD ALIAS MAPPING
  // This covers standard and verbose naming conventions across FOMM releases
  const tensorMapping = {
      // Images
      "image": sourceTensor,
      "source_image": sourceTensor,
      "source": sourceTensor,
      "input_image": sourceTensor,
      
      // Source Keypoints
      "source_keypoints": sourceKp,
      "source_keypoint_values": sourceKp,
      "kp_source": sourceKp,
      "source_kp": sourceKp,
      
      // Driving Keypoints
      "driving_keypoints": drivingKp,
      "driving_keypoint_values": drivingKp,
      "kp_driving": drivingKp,
      "driving_kp": drivingKp,
      
      // Jacobians
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
