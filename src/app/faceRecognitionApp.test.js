const { FaceRecognitionApp } = require('./faceRecognitionApp.js');
const { FaceEmbeddingStore } = require('../store/faceStore.js');

function createCanvasFactory() {
  return (width, height) => {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 100;
      data[i + 1] = 110;
      data[i + 2] = 120;
      data[i + 3] = 255;
    }
    const context = {
      fillStyle: '',
      fillRect: jest.fn(),
      drawImage: jest.fn(),
      setTransform: jest.fn(),
      getImageData: jest.fn(() => ({ data })),
    };
    const canvas = {
      width,
      height,
      getContext: jest.fn(() => context),
      __context: context,
    };
    return canvas;
  };
}

class FakeTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

describe('FaceRecognitionApp', () => {
  test('throws when no document is available for canvas creation', () => {
    expect(() => new FaceRecognitionApp({ documentRef: null })).toThrow('No document available');
  });

  test('loadModelSessions fails when runtime is missing', async () => {
    const createCanvas = createCanvasFactory();
    const app = new FaceRecognitionApp({ createCanvas });
    app.ort = null;
    await expect(app.loadModelSessions()).rejects.toThrow('ONNX Runtime not loaded');
  });

  test('initializes runtime and loads sessions', async () => {
    const createCanvas = createCanvasFactory();
    const detectorSession = { run: jest.fn() };
    const embedderSession = { run: jest.fn() };
    const createMock = jest.fn((url) => {
      if (url.includes('scrfd')) {
        return Promise.resolve(detectorSession);
      }
      return Promise.resolve(embedderSession);
    });
    const runtimeLoader = jest.fn(async () => ({
      ort: { Tensor: FakeTensor, InferenceSession: { create: createMock } },
      executionProviders: ['wasm'],
    }));
    const statusElement = { textContent: '' };
    const app = new FaceRecognitionApp({
      runtimeLoader,
      statusElement,
      createCanvas,
      detectorModelUrl: '/models/detector.onnx',
      embedderModelUrl: '/models/embedder.onnx',
    });
    await app.initialize();
    expect(runtimeLoader).toHaveBeenCalled();
    expect(createMock).toHaveBeenCalledWith('/models/detector.onnx', expect.any(Object));
    expect(statusElement.textContent).toContain('Loaded models');
  });

  test('detects, enrolls and identifies faces', async () => {
    const createCanvas = createCanvasFactory();
    const detectorSession = {
      run: jest.fn(async () => ({
        scores_8: { data: new Float32Array([-10, 10, -10, -10]) },
        boxes_8: { data: new Float32Array([0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]) },
        kps_8: { data: (() => {
          const arr = new Float32Array(40);
          for (let i = 0; i < 5; i += 1) {
            arr[10 + i * 2] = (i - 2) * 0.01;
            arr[10 + i * 2 + 1] = (i - 2) * 0.01;
          }
          return arr;
        })() },
      })),
    };
    const embedderSession = {
      run: jest.fn(async () => ({ fc1: { data: new Float32Array([1, 0, 0]) } })),
    };
    const store = new FaceEmbeddingStore({ matchThreshold: 0.2 });
    store.enroll('alice', [1, 0, 0]);

    const runtimeLoader = jest.fn(async () => ({
      ort: { Tensor: FakeTensor, InferenceSession: { create: jest.fn((url) => {
        if (url.includes('scrfd')) {
          return Promise.resolve(detectorSession);
        }
        return Promise.resolve(embedderSession);
      }) } },
      executionProviders: ['wasm'],
    }));

    const app = new FaceRecognitionApp({
      runtimeLoader,
      store,
      createCanvas,
      detectorInputSize: 16,
      alignedSize: 8,
      detectionScoreThreshold: 0.1,
      detectorStrides: [8],
    });

    await app.initialize();

    const sourceCanvas = createCanvas(16, 16);
    const detections = await app.detectFacesFromCanvas(sourceCanvas);
    expect(detections).toHaveLength(1);
    expect(detections[0].bbox[0]).toBeLessThan(detections[0].bbox[2]);

    await app.enrollFromCanvas(sourceCanvas, 'bob');
    const results = await app.identifyFromCanvas(sourceCanvas);
    expect(results[0].match.id).toBe('alice');
    expect(detectorSession.run).toHaveBeenCalled();
    expect(embedderSession.run).toHaveBeenCalled();
  });

  test('startCamera and captureVideoFrame interact with DOM APIs', async () => {
    const originalMediaDevices = navigator.mediaDevices;
    navigator.mediaDevices = { getUserMedia: jest.fn().mockResolvedValue('stream') };
    try {
      const createCanvas = createCanvasFactory();
      const video = document.createElement('video');
      video.play = jest.fn().mockResolvedValue();
      Object.defineProperty(video, 'videoWidth', { value: 320, writable: true });
      Object.defineProperty(video, 'videoHeight', { value: 240, writable: true });
      const app = new FaceRecognitionApp({ createCanvas, videoElement: video });
      const stream = await app.startCamera({ video: true });
      expect(stream).toBe('stream');
      expect(video.srcObject).toBe('stream');
      expect(video.play).toHaveBeenCalled();
      const frame = app.captureVideoFrame();
      expect(frame).toBe(app.captureCanvas);
      expect(app.captureContext.drawImage).toHaveBeenCalledWith(video, 0, 0, 320, 240);
    } finally {
      navigator.mediaDevices = originalMediaDevices;
    }
  });

  test('enrollFromCanvas throws when no detections', async () => {
    const createCanvas = createCanvasFactory();
    const detectorSession = {
      run: jest.fn(async () => ({
        scores_8: { data: new Float32Array([-10, -10, -10, -10]) },
        boxes_8: { data: new Float32Array(16) },
        kps_8: { data: new Float32Array(40) },
      })),
    };
    const embedderSession = {
      run: jest.fn(async () => ({ fc1: { data: new Float32Array([1, 0, 0]) } })),
    };
    const runtimeLoader = jest.fn(async () => ({
      ort: { Tensor: FakeTensor, InferenceSession: { create: jest.fn((url) => {
        if (url.includes('scrfd')) {
          return Promise.resolve(detectorSession);
        }
        return Promise.resolve(embedderSession);
      }) } },
      executionProviders: ['wasm'],
    }));
    const app = new FaceRecognitionApp({
      runtimeLoader,
      createCanvas,
      detectorInputSize: 16,
      detectorStrides: [8],
      detectionScoreThreshold: 0.5,
    });
    await app.initialize();
    const sourceCanvas = createCanvas(16, 16);
    await expect(app.enrollFromCanvas(sourceCanvas, 'nobody')).rejects.toThrow('No face detected');
  });

  test('throws when video element is missing', async () => {
    const createCanvas = createCanvasFactory();
    const app = new FaceRecognitionApp({ createCanvas });
    await expect(app.startCamera()).rejects.toThrow('Video element is not configured');
    expect(() => app.captureVideoFrame()).toThrow('Video element is not configured');
  });

  test('uses the browser document to create canvases by default', () => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = jest.fn(() => ({ drawImage: jest.fn() }));
    try {
      const app = new FaceRecognitionApp();
      expect(app.captureCanvas).toBeInstanceOf(HTMLCanvasElement);
    } finally {
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    }
  });

  test('embedAlignedCanvas supports raw tensors without a data property', async () => {
    const createCanvas = createCanvasFactory();
    const app = new FaceRecognitionApp({ createCanvas });
    app.ort = { Tensor: FakeTensor };
    app.embedderSession = {
      run: jest.fn(async () => ({
        [app.embedderOutputName]: new Float32Array([1, 0, 0]),
      })),
    };
    const aligned = createCanvas(app.alignedSize, app.alignedSize);
    const embedding = await app.embedAlignedCanvas(aligned);
    expect(app.embedderSession.run).toHaveBeenCalled();
    const norm = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));
    expect(norm).toBeCloseTo(1, 5);
  });
});
