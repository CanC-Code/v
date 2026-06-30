// motion-engine.js
// Finalized inference pipeline using dynamic graph interrogation.

let kpSession = null;
let genSession = null;

export function configureOrt() {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;
}

/**
 * Loads sessions and automatically logs the exact input/output names
 * required by the specific .onnx file version.
 */
export async function loadSessions(kpModelUrl, genModelUrl, executionProviders = ["wasm"]) {
  const kpDataUrl = kpModelUrl.replace('.onnx', '.data');
  const genDataUrl = genModelUrl.replace('.onnx', '.data');

  kpSession = await ort.InferenceSession.create(kpModelUrl, { 
    executionProviders,
    externalData: [{ path: kpModelUrl.split('/').pop().replace('.onnx', '.data'), data: kpDataUrl }]
  });
  
  genSession = await ort.InferenceSession.create(genModelUrl, { 
    executionProviders,
    externalData: [{ path: genModelUrl.split('/').pop().replace('.onnx', '.data'), data: genDataUrl }]
  });

  // CRITICAL: Log these to your browser console to see the exact strings
  console.log("KP Input Names:", kpSession.inputNames);
  console.log("Gen Input Names:", genSession.inputNames);
  
  return { kpSession, genSession };
}

export function isLoaded() {
  return !!(kpSession && genSession);
}

// Utility to convert canvas to Float32 Tensor
export function frameToTensor(canvas, size = 256) {
  const off = document.createElement("canvas");
  off.width = size; off.height = size;
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

async function detectKeypoints(frameTensor) {
  // Use the first available input name (usually 'image')
  const feeds = { [kpSession.inputNames[0]]: frameTensor };
  const results = await kpSession.run(feeds);
  
  // Return everything found so we can map it dynamically in runFrame
  return results; 
}

export async function runFrame(sourceTensor, sourceKpResult, drivingFrameTensor) {
  if (!isLoaded()) throw new Error("Models not loaded yet.");
  
  const drivingKpResult = await detectKeypoints(drivingFrameTensor);
  
  // DYNAMIC MAPPING:
  // We match our tensor data to the model's expected input names
  const feeds = {};
  
  // Helper to map tensors based on string content
  const mapTensor = (name, tensor) => { feeds[name] = tensor; };

  // Map Source
  mapTensor(genSession.inputNames.find(n => n.includes("source") || n.includes("image")), sourceTensor);
  
  // Map Keypoints (matching 'source_keypoint_values' or similar)
  mapTensor(genSession.inputNames.find(n => n.includes("source_keypoint")), sourceKpResult[Object.keys(sourceKpResult)[0]]);
  mapTensor(genSession.inputNames.find(n => n.includes("driving_keypoint")), drivingKpResult[Object.keys(drivingKpResult)[0]]);

  const results = await genSession.run(feeds);
  return results[genSession.outputNames[0]];
}
