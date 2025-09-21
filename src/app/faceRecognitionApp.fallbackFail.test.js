const { FaceRecognitionApp } = require('./faceRecognitionApp.js');

class FakeTensor { constructor() {} }

function createCanvasFactory() {
  return (w, h) => ({ width: w, height: h, getContext: () => ({ drawImage: () => {} }) });
}

describe('FaceRecognitionApp embedder EP fallback failure', () => {
  test('throws when both WebGPU and WASM embedder creation fail', async () => {
    const create = jest.fn(async (url, opts) => {
      // Detector succeeds
      if (url.includes('scrfd')) return {};
      // For embedder, always fail regardless of EP
      throw new Error('unsupported op');
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
    await expect(app.initialize()).rejects.toThrow('Failed to load embedder model');
    // Ensure it attempted both primary and wasm-only
    const embedderCalls = create.mock.calls.filter(([url]) => url.includes('arcfaceresnet100-11-int8.onnx'));
    expect(embedderCalls.length).toBe(2);
    expect(embedderCalls[0][1].executionProviders).toEqual(['webgpu', 'wasm']);
    expect(embedderCalls[1][1].executionProviders).toEqual(['wasm']);
  });
});
