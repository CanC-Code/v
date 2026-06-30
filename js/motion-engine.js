// motion-engine.js
// Finalized inference pipeline for Qualcomm FOMM ONNX assets.

export const IO_CONFIG = {
  // Retained structurally for external UI compatibility, but internal 
  // execution now dynamically binds to the graph's actual requirement strings.
  kpDetector: {
    inputName: "image",
    outputKeypoints: "keypoints",
    outputJacobian: "jacobian", 
  },
  generator: {
    inputSource: "image", // Corrected default mapping from 'source_image' to 'image'
    inputKpSource: "source_keypoints",
    inputKpDriving: "driving_keypoints",
    inputJacobianSource: "source_jacobian",
    inputJacobianDriving: "driving_jacobian",
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
  // 1. Verify availability first to debug 404s
  await verifyFile(kpModelUrl);
  await verifyFile(genModelUrl);
  
  // Extract expected external data internal identifiers and target URLs
  const kpDataFileName = kpModelUrl.split('/').pop().replace('.onnx', '.data');
  const genDataFileName = genModelUrl.split('/').pop().replace('.onnx', '.data');

  const kpDataUrl = kpModelUrl.replace('.onnx', '.data');
  const genDataUrl = genModelUrl.replace('.onnx', '.data');

  // Verify external weight data files are available before ORT loads
  await verifyFile(kpDataUrl);
  await verifyFile(genDataUrl);

  // 2. Create sessions with externalData configured to prevent Emscripten FileSystem mount errors
  kpSession = await ort.InferenceSession.create(kpModelUrl, { 
    executionProviders,
    externalData: [
      { path: kpDataFileName, data: kpDataUrl }
    ]
  });
  
  genSession = await ort.InferenceSession.create(genModelUrl, { 
    executionProviders,
    externalData: [
      { path: genDataFileName, data: genDataUrl }
    ]
  });

  // Debug: Log the loaded layers
  console.log("Detector Inputs:", kpSession.inputNames);
  console.log("Detector Outputs:", kpSession.outputNames);
  console.log("Generator Inputs:", genSession.inputNames);
  console.log("Generator Outputs:", genSession.outputNames);
  
  return { kpSession, genSession };
}

export function isLoaded() {
  return !!(kpSession && genSession);
}

export function frameToTensor(canvas, size = IO_CONFIG.frameSize) {
  const off = document.createElement("canvas");
  off.width = size;
  off.height = size;
  const ctx = off.getContext("2d");
  ctx.drawImage(canvas, 0, 0, size, size);
  
  const { data } = ctx.getImageData(0, 0, size, size);

  const floatData = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    floatData[i] = data[i * 4] / 255;
    floatData[size * size + i] = data[i * 4 + 1] / 255;
    floatData[2 * size * size + i] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

export function tensorToImageData(tensor, size = IO_CONFIG.frameSize) {
  const data = tensor.data;
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    out[i * 4] = clamp255(data[i] * 255);
    out[i * 4 + 1] = clamp255(data[size * size + i] * 255);
    out[i * 4 + 2] = clamp255(data[2 * size * size + i] * 255);
    out[i * 4 + 3] = 255;
  }
  return new ImageData(out, size, size);
}

function clamp255(v) { return Math.max(0, Math.min(255, v)); }

// DYNAMIC INPUT MAPPING
// Guarantees we fulfill the ONNX graph requirements regardless of internal metadata naming conventions.
function buildFeeds(session, mapping) {
    const feeds = {};
    for (const expectedName of session.inputNames) {
        // Ensure we do not push null values (like absent Jacobians) which crash execution.
        if (mapping[expectedName] !== undefined && mapping[expectedName] !== null) {
            feeds[expectedName] = mapping[expectedName];
        } else {
            console.warn(`Model expects input '${expectedName}' but no valid tensor was mapped.`);
        }
    }
    return feeds;
}

async function detectKeypoints(frameTensor) {
  // Determine correct input string dynamically
  const inputName = kpSession.inputNames.includes("image") ? "image" : 
                   (kpSession.inputNames.includes("source") ? "source" : IO_CONFIG.kpDetector.inputName);
                   
  const feeds = { [inputName]: frameTensor };
  const results = await kpSession.run(feeds);
  
  // Extract dynamically based on actual output names
  const kpKey = kpSession.outputNames.find(n => n.includes("keypoint") || n === "kp") || IO_CONFIG.kpDetector.outputKeypoints;
  const jacKey = kpSession.outputNames.find(n => n.includes("jacobian") || n === "jac");

  const kp = results[kpKey];
  const jac = jacKey ? results[jacKey] : null;
  
  if (!kp) {
      console.error("Available tensors in KP:", Object.keys(results));
      throw new Error(`Missing tensor: ${kpKey}`);
  }
  return { kp, jac };
}

export async function runFrame(sourceTensor, sourceKp, sourceJac, drivingFrameTensor) {
  if (!isLoaded()) throw new Error("Models not loaded yet.");
  
  const { kp: drivingKp, jac: drivingJac } = await detectKeypoints(drivingFrameTensor);
  
  // Map all known permutations of ONNX inputs to our local variables.
  // This resolves strict 'image' vs 'source_image' missing feeds errors.
  const tensorMapping = {
      // Source Image Variations
      "image": sourceTensor,
      "source_image": sourceTensor,
      "source": sourceTensor,
      
      // Keypoint Variations
      "source_keypoints": sourceKp,
      "kp_source": sourceKp,
      "driving_keypoints": drivingKp,
      "kp_driving": drivingKp,
      
      // Jacobian Variations
      "source_jacobian": sourceJac,
      "jac_source": sourceJac,
      "driving_jacobian": drivingJac,
      "jac_driving": drivingJac
  };

  // Construct final feeds matching the loaded model precisely
  const feeds = buildFeeds(genSession, tensorMapping);

  const results = await genSession.run(feeds);
  
  // Determine correct output key
  const outKey = genSession.outputNames.find(n => n.includes("image") || n === "prediction") || IO_CONFIG.generator.output;
  const out = results[outKey];
  
  if (!out) {
      console.error("Available tensors in Generator:", Object.keys(results));
      throw new Error(`Missing output tensor: ${outKey}`);
  }
  return out;
}

export async function computeSourceKeypoints(sourceTensor) {
  return detectKeypoints(sourceTensor);
}
