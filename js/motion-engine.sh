// motion-engine.js
// Finalized inference pipeline for Qualcomm FOMM ONNX assets.

export const IO_CONFIG = {
  kpDetector: {
    inputName: "source",
    outputKeypoints: "kp",
    outputJacobian: "jac",
  },
  generator: {
    inputSource: "source",
    inputKpSource: "kp_source",
    inputKpDriving: "kp_driving",
    inputJacobianSource: "jac_source",
    inputJacobianDriving: "jac_driving",
    output: "prediction",
  },
  frameSize: 256,
};

let kpSession = null;
let genSession = null;

export function configureOrt() {
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;
}

export async function loadSessions(kpModelBytes, genModelBytes, executionProviders = ["wasm"]) {
  kpSession = await ort.InferenceSession.create(kpModelBytes, { executionProviders });
  genSession = await ort.InferenceSession.create(genModelBytes, { executionProviders });
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

async function detectKeypoints(frameTensor) {
  const feeds = { [IO_CONFIG.kpDetector.inputName]: frameTensor };
  const results = await kpSession.run(feeds);
  const kp = results[IO_CONFIG.kpDetector.outputKeypoints];
  const jac = results[IO_CONFIG.kpDetector.outputJacobian] || null;
  if (!kp) throw new Error(`Missing tensor: ${IO_CONFIG.kpDetector.outputKeypoints}`);
  return { kp, jac };
}

export async function runFrame(sourceTensor, sourceKp, sourceJac, drivingFrameTensor) {
  if (!isLoaded()) throw new Error("Models not loaded yet.");

  const { kp: drivingKp, jac: drivingJac } = await detectKeypoints(drivingFrameTensor);

  const feeds = {
    [IO_CONFIG.generator.inputSource]: sourceTensor,
    [IO_CONFIG.generator.inputKpSource]: sourceKp,
    [IO_CONFIG.generator.inputKpDriving]: drivingKp,
  };
  if (sourceJac && drivingJac) {
    feeds[IO_CONFIG.generator.inputJacobianSource] = sourceJac;
    feeds[IO_CONFIG.generator.inputJacobianDriving] = drivingJac;
  }

  const results = await genSession.run(feeds);
  const out = results[IO_CONFIG.generator.output];
  if (!out) throw new Error(`Missing output tensor: ${IO_CONFIG.generator.output}`);
  return out;
}

export async function computeSourceKeypoints(sourceTensor) {
  return detectKeypoints(sourceTensor);
}
