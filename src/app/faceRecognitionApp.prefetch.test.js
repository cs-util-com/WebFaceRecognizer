const { FaceRecognitionApp } = require('./faceRecognitionApp.js');

class FakeTensor { constructor() {} }

function createCanvasFactory() {
  return (w, h) => ({ width: w, height: h, getContext: () => ({ drawImage: () => {} }) });
}

describe('FaceRecognitionApp embedder prefetch optimization', () => {
  let originalFetch;
  beforeAll(() => {
    originalFetch = global.fetch;
  });
  afterAll(() => {
    global.fetch = originalFetch;
  });

  test('prefetch succeeds: passes Uint8Array to session.create and no fallback', async () => {
    // Mock fetch to return bytes
    global.fetch = jest.fn(async () => ({ ok: true, arrayBuffer: async () => new Uint8Array([1,2,3]).buffer }));
    const create = jest.fn(async () => {
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
      prefetchEmbedderModel: true,
    });
    await expect(app.initialize()).resolves.toBeUndefined();
    // Find the embedder call
    const embedderCall = create.mock.calls.find(([arg]) =>
      typeof arg !== 'string' // prefetch passes Uint8Array, not URL string
    );
    expect(embedderCall).toBeDefined();
    const [modelArg, opts] = embedderCall;
    expect(modelArg).toBeInstanceOf(Uint8Array);
    expect(opts.executionProviders).toEqual(['webgpu', 'wasm']);
    // Ensure only one embedder attempt
    const embedderCalls = create.mock.calls.filter(([arg]) => typeof arg !== 'string');
    expect(embedderCalls.length).toBe(1);
  });

  test('prefetch fails: uses URL, falls back from WebGPU to WASM', async () => {
    // Mock fetch to fail
    global.fetch = jest.fn(async () => { throw new Error('network'); });
    const create = jest.fn(async (url, opts) => {
      if (url.includes && url.includes('scrfd')) return {}; // detector ok
      // For embedder: fail when trying webgpu path, succeed for wasm-only
      if (opts && Array.isArray(opts.executionProviders) && opts.executionProviders.includes('webgpu')) {
        throw new Error('unsupported op');
      }
      return {};
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
      prefetchEmbedderModel: true,
    });
    await expect(app.initialize()).resolves.toBeUndefined();
    // First embedder attempt used URL string and webgpu+wasm EP
    const firstEmbedder = create.mock.calls.find(([arg, o]) => typeof arg === 'string' && o.executionProviders.includes('webgpu'));
    expect(firstEmbedder).toBeDefined();
    // Second embedder attempt used URL string and wasm-only EP
    const secondEmbedder = create.mock.calls.find(([arg, o]) => typeof arg === 'string' && JSON.stringify(o.executionProviders) === JSON.stringify(['wasm']));
    expect(secondEmbedder).toBeDefined();
  });
});
