const { FaceRecognitionApp } = require('./faceRecognitionApp.js');

class FakeTensor { constructor() {} }

function createCanvasFactory() {
  return (w, h) => ({ width: w, height: h, getContext: () => ({ drawImage: () => {} }) });
}

describe('FaceRecognitionApp embedder EP fallback', () => {
  test('retries embedder creation with WASM when WebGPU path fails', async () => {
    const create = jest.fn(async (url, opts) => {
      // Detector succeeds immediately
      if (url.includes('scrfd')) return {};
      // Embedder: fail when webgpu is present, succeed for wasm-only
      if (Array.isArray(opts.executionProviders) && opts.executionProviders.includes('webgpu')) {
        throw new Error('WebGPU unsupported op');
      }
      if (Array.isArray(opts.executionProviders) && opts.executionProviders.length === 1 && opts.executionProviders[0] === 'wasm') {
        return {};
      }
      throw new Error('Unexpected EP config');
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
    // Ensure create called twice for embedder: one failure (with webgpu) then success (wasm-only)
    const embedderCalls = create.mock.calls.filter(([url]) => url.includes('arcfaceresnet100-11-int8.onnx'));
    expect(embedderCalls.length).toBe(2);
    expect(embedderCalls[0][1].executionProviders).toEqual(['webgpu', 'wasm']);
    expect(embedderCalls[1][1].executionProviders).toEqual(['wasm']);
  });
});
