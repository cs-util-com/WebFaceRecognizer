import { FaceRecognitionApp } from './faceRecognitionApp.js';

function createDoc() {
  return {
    createElement: (tag) => {
      if (tag !== 'canvas') throw new Error('expected canvas');
      const ctx = {
        drawImage: jest.fn(),
        fillRect: jest.fn(),
        setTransform: jest.fn(),
        getImageData: jest.fn(() => ({ data: new Uint8ClampedArray(4) })),
      };
      return {
        width: 0,
        height: 0,
        getContext: () => ctx,
        __ctx: ctx,
      };
    },
  };
}

test('identifyFromImageBitmap delegates to identifyFromCanvas', async () => {
  const app = new FaceRecognitionApp({ documentRef: createDoc(), runtimeLoader: async () => ({ ort: { Tensor: function(){} }, executionProviders: ['wasm'] }) });
  app.identifyFromCanvas = jest.fn().mockResolvedValue([]);
  const img = { width: 10, height: 20 };
  const res = await app.identifyFromImageBitmap(img);
  expect(res).toEqual([]);
  expect(app.identifyFromCanvas).toHaveBeenCalled();
});
