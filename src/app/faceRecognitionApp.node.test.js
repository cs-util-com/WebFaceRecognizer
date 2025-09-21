/** @jest-environment node */

const { FaceRecognitionApp } = require('./faceRecognitionApp.js');

describe('FaceRecognitionApp (node environment)', () => {
  test('uses undefined document reference when not in a browser', () => {
    const createCanvas = jest.fn((width, height) => ({
      width,
      height,
      getContext: jest.fn(() => ({ drawImage: jest.fn(), fillRect: jest.fn() })),
    }));
    const app = new FaceRecognitionApp({ createCanvas });
    expect(app.document).toBeUndefined();
    expect(createCanvas).toHaveBeenCalledWith(app.detectorInputSize, app.detectorInputSize);
  });
});
