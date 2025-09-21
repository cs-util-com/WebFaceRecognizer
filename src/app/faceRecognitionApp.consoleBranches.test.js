const { FaceRecognitionApp } = require('./faceRecognitionApp.js');

class FakeTensor { constructor() {} }

function createCanvasFactory() {
  return (w, h) => ({ width: w, height: h, getContext: () => ({ drawImage: () => {} }) });
}

describe('FaceRecognitionApp console guarded branches', () => {
  let originalConsole;
  beforeEach(() => {
    originalConsole = global.console;
  });
  afterEach(() => {
    global.console = originalConsole;
  });

  test('initialize and load sessions when console is undefined (covers else branches)', async () => {
    // Remove console to exercise guarded branches where console.* are not available
    // eslint-disable-next-line no-global-assign
    console = undefined;
  const create = jest.fn(async () => ({}));
    const runtimeLoader = async () => ({
      ort: { Tensor: FakeTensor, InferenceSession: { create } },
      executionProviders: ['wasm']
    });
    const app = new FaceRecognitionApp({
      runtimeLoader,
      createCanvas: createCanvasFactory(),
      detectorModelUrl: '/models/scrfd_2.5g_bnkps.onnx',
      embedderModelUrl: '/models/arcfaceresnet100-11-int8.onnx',
      prefetchEmbedderModel: false,
    });
    await expect(app.initialize()).resolves.toBeUndefined();
    // Two session creations: detector + embedder
    expect(create).toHaveBeenCalledTimes(2);
  });
});
