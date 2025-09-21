const { FaceRecognitionApp } = require('./faceRecognitionApp.js');

class FakeTensor { constructor() {} }

function createCanvasFactory() {
  return (w, h) => ({ width: w, height: h, getContext: () => ({ drawImage: () => {} }) });
}

describe('FaceRecognitionApp embedder creation succeeds on WebGPU path', () => {
  test('creates embedder with ["webgpu","wasm"] without falling back', async () => {
    const create = jest.fn(async (url, opts) => {
      return {}; // succeed for both detector and embedder
    });
    const runtimeLoader = async () => ({
      ort: { Tensor: FakeTensor, InferenceSession: { create } },
      executionProviders: ['webgpu']
    });
    const app = new FaceRecognitionApp({
      runtimeLoader,
      createCanvas: createCanvasFactory(),
      detectorModelUrl: '/models/scrfd_2.5g_bnkps.onnx',
      embedderModelUrl: '/models/arcfaceresnet100-11-int8.onnx',
    });
    await expect(app.initialize()).resolves.toBeUndefined();
    const calls = create.mock.calls;
    // Last call corresponds to embedder
    const embedderCall = calls.find(([url]) => url.includes('arcfaceresnet100-11-int8.onnx'));
    expect(embedderCall).toBeDefined();
    expect(embedderCall[1].executionProviders).toEqual(['webgpu', 'wasm']);
    // Ensure only one attempt for embedder (no fallback)
    const embedderCalls = calls.filter(([url]) => url.includes('arcfaceresnet100-11-int8.onnx'));
    expect(embedderCalls.length).toBe(1);
  });
});
