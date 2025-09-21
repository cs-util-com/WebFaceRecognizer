const { FaceRecognitionApp } = require('./faceRecognitionApp.js');

class FakeTensor { constructor() {} }

describe('FaceRecognitionApp model loading errors', () => {
  function createCanvasFactory() {
    return (w, h) => ({ width: w, height: h, getContext: () => ({ drawImage: () => {} }) });
  }

  test('throws helpful error when detector model 404s', async () => {
    const create = jest.fn(async (url) => {
      if (url.includes('missing.onnx')) {
        throw new Error('failed to load external data file: /models/missing.onnx');
      }
      return {};
    });
    const runtimeLoader = async () => ({ ort: { Tensor: FakeTensor, InferenceSession: { create } }, executionProviders: ['wasm'] });
    const app = new FaceRecognitionApp({
      runtimeLoader,
      createCanvas: createCanvasFactory(),
      detectorModelUrl: '/models/missing.onnx',
      embedderModelUrl: '/models/arcface.onnx',
    });
    await expect(app.initialize()).rejects.toThrow('Failed to load detector model at /models/missing.onnx');
  });

  test('throws helpful error when embedder model fails', async () => {
    const create = jest.fn(async (url) => {
      if (url.includes('scrfd')) return {};
      throw new Error('some other error');
    });
    const runtimeLoader = async () => ({ ort: { Tensor: FakeTensor, InferenceSession: { create } }, executionProviders: ['wasm'] });
    const app = new FaceRecognitionApp({
      runtimeLoader,
      createCanvas: createCanvasFactory(),
      detectorModelUrl: '/models/scrfd.onnx',
      embedderModelUrl: '/models/arcface_missing.onnx',
    });
    await expect(app.initialize()).rejects.toThrow('Failed to load embedder model at /models/arcface_missing.onnx');
  });
});
