// motion-engine.js
// Generic motion-transfer inference pipeline using ONNX Runtime Web.
//
// HONESTY NOTE: Motion-transfer model exports (e.g. First Order Motion Model /
// Thin-Plate-Spline-Motion-Model ports to ONNX) are not standardized — input/output
// tensor names and the exact keypoint representation vary by who exported them.
// The defaults below match the most common community ONNX exports of FOMM, but
// you MUST verify them against whatever model you actually load (e.g. with
// https://netron.app) and adjust `IO_CONFIG` if they don't match. Inference will
// throw a clear error if a tensor name is wrong rather than silently producing
// garbage.

export const IO_CONFIG = {
  kpDetector: {
    inputName: "source",        // image tensor in -> keypoints out
    outputKeypoints: "keypoints",
    outputJacobian: "jacobian",
  },
  generator: {
    inputSource: "source",
    inputKpSource: "kp_source",
    inputKpDriving: "kp_driving",
    inputJacobianSource: "jacobian_source",
    inputJacobianDriving: "jacobian_driving",
    output: "prediction",
  },
  frameSize: 256, // most FOMM-family models expect 256x256 RGB, normalized 0-1
};

let kpSession = null;
let genSession = null;

export function configureOrt() {
  // Run fully on-device: WASM backend with SIMD + multithreading where available.
  // WebGPU is used automatically if ort detects support and the model supports it.
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

/**
 * Convert a canvas/video frame into a normalized NCHW Float32 ONNX tensor.
 */
export function frameToTensor(canvas, size = IO_CONFIG.frameSize) {
  const off = document.createElement("canvas");
  off.width = size;
  off.height = size;
  const ctx = off.getContext("2d");
  ctx.drawImage(canvas, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);

  const floatData = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    floatData[i] = data[i * 4] / 255;                       // R
    floatData[size * size + i] = data[i * 4 + 1] / 255;      // G
    floatData[2 * size * size + i] = data[i * 4 + 2] / 255;  // B
  }

  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

/**
 * Convert an NCHW Float32 tensor back to an ImageData for drawing to canvas.
 */
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

function clamp255(v) {
  return Math.max(0, Math.min(255, v));
}

async function detectKeypoints(frameTensor) {
  const feeds = { [IO_CONFIG.kpDetector.inputName]: frameTensor };
  const results = await kpSession.run(feeds);
  const kp = results[IO_CONFIG.kpDetector.outputKeypoints];
  const jac = results[IO_CONFIG.kpDetector.outputJacobian] || null;
  if (!kp) {
    throw new Error(
      `Keypoint detector did not return tensor "${IO_CONFIG.kpDetector.outputKeypoints}". ` +
      `Check your model's actual output names (inspect with netron.app) and update IO_CONFIG.`
    );
  }
  return { kp, jac };
}

/**
 * Run one frame of motion transfer.
 * sourceTensor: tensor of the still subject image (computed once, reused every frame)
 * sourceKp/sourceJac: keypoints of the source image (computed once)
 * drivingFrameTensor: tensor of the current driving-video frame
 * Returns: ort.Tensor of the generated frame
 */
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
  if (!out) {
    throw new Error(
      `Generator did not return tensor "${IO_CONFIG.generator.output}". ` +
      `Check your model's actual output name and update IO_CONFIG.`
    );
  }
  return out;
}

export async function computeSourceKeypoints(sourceTensor) {
  return detectKeypoints(sourceTensor);
}
