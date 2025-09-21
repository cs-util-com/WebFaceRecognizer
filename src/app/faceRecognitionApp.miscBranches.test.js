const { FaceRecognitionApp } = require('./faceRecognitionApp.js');

class FakeTensor { constructor() {} }

function createCanvasFactory() {
  return (w, h) => ({ width: w, height: h, getContext: () => ({ drawImage: () => {} }) });
}

describe('FaceRecognitionApp misc branch coverage', () => {
  test('initialize works with console methods undefined (no time/info)', async () => {
    const originalConsole = global.console;
    // Define minimal console missing time/timeEnd/info to take false branches
    global.console = { debug: jest.fn() };
    try {
      const create = jest.fn(async () => ({}));
      const runtimeLoader = async () => ({ ort: { Tensor: FakeTensor, InferenceSession: { create } }, executionProviders: ['wasm'] });
      const app = new FaceRecognitionApp({ runtimeLoader, createCanvas: createCanvasFactory(), documentRef: document });
      await expect(app.initialize()).resolves.toBeUndefined();
    } finally {
      global.console = originalConsole;
    }
  });
});
