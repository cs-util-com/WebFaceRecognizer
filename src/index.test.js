global.__createFakeApp = () => { throw new Error('Factory not set'); };

jest.mock('./app/faceRecognitionApp.js', () => ({
  FaceRecognitionApp: jest.fn((config) => global.__createFakeApp(config)),
}));

const { FaceRecognitionApp } = require('./app/faceRecognitionApp.js');
const { bootstrapFaceRecognitionApp, createDefaultAppConfig, drawDetections, renderMatchList } = require('./index.js');

function createCanvasStub(width = 10, height = 10) {
  const context = {
    clearRect: jest.fn(),
    strokeRect: jest.fn(),
    fillRect: jest.fn(),
    fillText: jest.fn(),
    measureText: jest.fn(() => ({ width: 40 })),
    beginPath: jest.fn(),
    arc: jest.fn(),
    fill: jest.fn(),
    getImageData: jest.fn(() => ({ data: new Uint8ClampedArray(width * height * 4) })),
  };
  const canvas = document.createElement('canvas');
  Object.defineProperty(canvas, 'width', { value: width, writable: true });
  Object.defineProperty(canvas, 'height', { value: height, writable: true });
  canvas.getContext = jest.fn(() => context);
  canvas.__context = context;
  return canvas;
}

describe('UI helpers', () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <video id="camera"></video>
      <canvas id="overlay"></canvas>
      <p id="status"></p>
      <button id="start"></button>
      <button id="identify"></button>
      <button id="enroll"></button>
      <input id="upload" type="file" />
      <input id="identity" />
      <ul id="results"></ul>
    `;
  });

  test('createDefaultAppConfig selects required elements', () => {
    const config = createDefaultAppConfig(document);
    expect(config.videoElement.id).toBe('camera');
    expect(config.overlayCanvas.id).toBe('overlay');
    expect(config.statusElement.id).toBe('status');
  });

  test('createDefaultAppConfig throws when elements are missing', () => {
    document.getElementById('camera').remove();
    expect(() => createDefaultAppConfig(document)).toThrow('camera');
  });

  test('drawDetections renders boxes and keypoints', () => {
    const canvas = createCanvasStub(100, 100);
    const detection = { bbox: [10, 10, 50, 50], keypoints: [[20, 20], [30, 30], [40, 40], [50, 50], [60, 60]] };
    drawDetections(canvas, [detection], [{ match: { id: 'alice', score: 0.95 } }]);
    const ctx = canvas.__context;
    expect(ctx.strokeRect).toHaveBeenCalled();
    expect(ctx.arc).toHaveBeenCalledTimes(5);
    drawDetections(canvas, [detection], []);
    drawDetections(null, [detection]);
  });

  test('drawDetections without matches does not render labels', () => {
    const canvas = createCanvasStub(50, 50);
    const detection = { bbox: [5, 5, 20, 20], keypoints: [[6, 6], [7, 7], [8, 8], [9, 9], [10, 10]] };
    drawDetections(canvas, [detection], [null]);
    const ctx = canvas.__context;
    // Should still draw box and keypoints, but no label fillText call
    expect(ctx.strokeRect).toHaveBeenCalled();
    expect(ctx.arc).toHaveBeenCalledTimes(5);
  });

  test('renderMatchList populates DOM', () => {
    const list = document.getElementById('results');
    renderMatchList(list, [
      { detection: { bbox: [0, 0, 10, 10] }, match: { id: 'alice', score: 0.9 } },
      { detection: { bbox: [0, 0, 10, 10] }, match: null },
    ]);
    expect(list.children).toHaveLength(2);
    expect(list.textContent).toContain('alice');
    renderMatchList(null, []);
  });

  test('renderMatchList renders unknown entries', () => {
    const list = document.getElementById('results');
    renderMatchList(list, [
      { detection: { bbox: [1, 2, 3, 4] }, match: null },
    ]);
    expect(list.textContent).toContain('Unknown face');
  });

  test('attaches bootstrap to window in browser-like environment', () => {
    // In jsdom environment, the module should attach bootstrap to window
    jest.isolateModules(() => {
      require('./index.js');
    });
    expect(typeof window.bootstrapFaceRecognitionApp).toBe('function');
  });

  test('bootstrap wires event handlers to the fake app', async () => {
    const overlayCanvas = createCanvasStub(100, 100);
    document.getElementById('overlay').replaceWith(overlayCanvas);
    overlayCanvas.id = 'overlay';

    const frameCanvas = createCanvasStub(120, 80);

    const startButton = document.getElementById('start');
    const enrollButton = document.getElementById('enroll');
    const identifyButton = document.getElementById('identify');
    const startListener = jest.spyOn(startButton, 'addEventListener');
    const enrollListener = jest.spyOn(enrollButton, 'addEventListener');
    const identifyListener = jest.spyOn(identifyButton, 'addEventListener');
  const uploadInput = document.getElementById('upload');
  const uploadListener = jest.spyOn(uploadInput, 'addEventListener');

    const identifyResult = [{
      detection: { bbox: [0, 0, 20, 20], keypoints: [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]] },
      match: { id: 'alice', score: 0.88 },
    }];

    global.__createFakeApp = () => ({
      initialize: jest.fn().mockResolvedValue(),
      startCamera: jest.fn().mockResolvedValue(),
      enrollFromCanvas: jest.fn().mockResolvedValue({}),
      identifyFromCanvas: jest.fn().mockResolvedValue(identifyResult),
      captureVideoFrame: jest.fn(() => frameCanvas),
      updateStatus: jest.fn(),
    });

    await bootstrapFaceRecognitionApp({ documentRef: document, autoInit: false });
    expect(FaceRecognitionApp).toHaveBeenCalled();

    const fakeInstance = FaceRecognitionApp.mock.results[0].value;
    const startHandler = startListener.mock.calls[0][1];
    const enrollHandler = enrollListener.mock.calls[0][1];
    const identifyHandler = identifyListener.mock.calls[0][1];
  const uploadHandler = uploadListener.mock.calls[0][1];

    fakeInstance.startCamera.mockRejectedValueOnce(new Error('camera fail'));
    await startHandler(new Event('click'));
    await startHandler(new Event('click'));

    const identityInput = document.getElementById('identity');
    identityInput.value = '';
    await enrollHandler(new Event('click'));

    identityInput.value = 'alice';
    fakeInstance.captureVideoFrame.mockImplementationOnce(() => { throw new Error('capture fail'); });
    fakeInstance.enrollFromCanvas.mockRejectedValueOnce(new Error('fail'));
    await enrollHandler(new Event('click'));
    await enrollHandler(new Event('click'));

    fakeInstance.identifyFromCanvas.mockRejectedValueOnce(new Error('identify fail'));
    await identifyHandler(new Event('click'));
    await identifyHandler(new Event('click'));

    expect(fakeInstance.startCamera).toHaveBeenCalled();
    expect(fakeInstance.updateStatus).toHaveBeenCalledWith('Camera started');
    expect(fakeInstance.enrollFromCanvas).toHaveBeenCalledWith(frameCanvas, 'alice', expect.any(Object));
    expect(fakeInstance.identifyFromCanvas).toHaveBeenCalledWith(frameCanvas);
    const list = document.getElementById('results');
    expect(list.textContent).toContain('alice');

    // Simulate upload flow: provide a fake image decode path
    const fileEvent = { target: { files: [new Blob()] } };
    global.fetch = jest.fn().mockResolvedValue({ ok: true, blob: () => Promise.resolve(new Blob()) });
    global.createImageBitmap = jest.fn().mockResolvedValue({ width: 64, height: 48 });
    await uploadHandler(fileEvent);
    expect(fakeInstance.identifyFromCanvas).toHaveBeenCalled();

    // Early return when no file selected
    await uploadHandler({ target: { files: [] } });

    // Error path: createImageBitmap fails
    global.createImageBitmap = jest.fn().mockRejectedValue(new Error('decode fail'));
    await uploadHandler({ target: { files: [new Blob()] } });
    expect(fakeInstance.updateStatus).toHaveBeenCalledWith('decode fail');
  });

  test('bootstrap initializes models when autoInit is enabled', async () => {
    global.__createFakeApp = () => ({
      initialize: jest.fn().mockResolvedValue(),
      startCamera: jest.fn(),
      enrollFromCanvas: jest.fn(),
      identifyFromCanvas: jest.fn(),
      captureVideoFrame: jest.fn(() => createCanvasStub(10, 10)),
      updateStatus: jest.fn(),
    });
    const app = await bootstrapFaceRecognitionApp({ documentRef: document });
    expect(app.initialize).toHaveBeenCalled();
  });

  test('bootstrap applies model URL overrides from query params', async () => {
    const detector = '/models/override-det.onnx';
    const embedder = '/models/override-emb.onnx';
    const stubDoc = {
      getElementById: document.getElementById.bind(document),
      location: { href: `https://example.test/?detector=${encodeURIComponent(detector)}&embedder=${encodeURIComponent(embedder)}` },
    };

    let capturedConfig;
    global.__createFakeApp = (config) => {
      capturedConfig = config;
      return {
        initialize: jest.fn(),
        startCamera: jest.fn(),
        enrollFromCanvas: jest.fn(),
        identifyFromCanvas: jest.fn(),
        captureVideoFrame: jest.fn(() => createCanvasStub(10, 10)),
        updateStatus: jest.fn(),
      };
    };

    await bootstrapFaceRecognitionApp({ documentRef: stubDoc, autoInit: false });
    expect(capturedConfig.detectorModelUrl).toBe(detector);
    expect(capturedConfig.embedderModelUrl).toBe(embedder);
  });

  test('bootstrap applies only detector override when embedder missing', async () => {
    const detector = '/models/override-det.onnx';
    const stubDoc = {
      getElementById: document.getElementById.bind(document),
      location: { href: `https://example.test/?detector=${encodeURIComponent(detector)}` },
    };

    let capturedConfig;
    global.__createFakeApp = (config) => {
      capturedConfig = config;
      return {
        initialize: jest.fn(),
        startCamera: jest.fn(),
        enrollFromCanvas: jest.fn(),
        identifyFromCanvas: jest.fn(),
        captureVideoFrame: jest.fn(() => createCanvasStub(10, 10)),
        updateStatus: jest.fn(),
      };
    };

    await bootstrapFaceRecognitionApp({ documentRef: stubDoc, autoInit: false });
    expect(capturedConfig.detectorModelUrl).toBe(detector);
    expect(capturedConfig.embedderModelUrl).toBeUndefined();
  });

  test('bootstrap applies only embedder override when detector missing', async () => {
    const embedder = '/models/override-emb.onnx';
    const stubDoc = {
      getElementById: document.getElementById.bind(document),
      location: { href: `https://example.test/?embedder=${encodeURIComponent(embedder)}` },
    };

    let capturedConfig;
    global.__createFakeApp = (config) => {
      capturedConfig = config;
      return {
        initialize: jest.fn(),
        startCamera: jest.fn(),
        enrollFromCanvas: jest.fn(),
        identifyFromCanvas: jest.fn(),
        captureVideoFrame: jest.fn(() => createCanvasStub(10, 10)),
        updateStatus: jest.fn(),
      };
    };

    await bootstrapFaceRecognitionApp({ documentRef: stubDoc, autoInit: false });
    expect(capturedConfig.detectorModelUrl).toBeUndefined();
    expect(capturedConfig.embedderModelUrl).toBe(embedder);
  });

  test('bootstrap applies advanced embedder overrides including types and normalization', async () => {
    const params = new URLSearchParams({
      embedder: '/models/int8.onnx',
      embedderInputType: 'uint8',
      embedderInputName: 'input0',
      embedderOutputName: 'output0',
      normalizeMean: '0',
      normalizeScale: '0.5',
    });
    const stubDoc = {
      getElementById: document.getElementById.bind(document),
      location: { href: `https://example.test/?${params.toString()}` },
    };

    let capturedConfig;
    global.__createFakeApp = (config) => {
      capturedConfig = config;
      return {
        initialize: jest.fn(),
        startCamera: jest.fn(),
        enrollFromCanvas: jest.fn(),
        identifyFromCanvas: jest.fn(),
        captureVideoFrame: jest.fn(() => createCanvasStub(10, 10)),
        updateStatus: jest.fn(),
      };
    };

    await bootstrapFaceRecognitionApp({ documentRef: stubDoc, autoInit: false });
    expect(capturedConfig.embedderModelUrl).toBe('/models/int8.onnx');
    expect(capturedConfig.embedderInputType).toBe('uint8');
    expect(capturedConfig.embedderInputName).toBe('input0');
    expect(capturedConfig.embedderOutputName).toBe('output0');
    expect(capturedConfig.normalizeMean).toBe(0);
    expect(capturedConfig.normalizeScale).toBe(0.5);
  });

  test('bootstrap ignores invalid normalization numbers', async () => {
    const params = new URLSearchParams({
      normalizeMean: 'NaN',
      normalizeScale: 'foo',
    });
    const stubDoc = {
      getElementById: document.getElementById.bind(document),
      location: { href: `https://example.test/?${params.toString()}` },
    };

    let capturedConfig;
    global.__createFakeApp = (config) => {
      capturedConfig = config;
      return {
        initialize: jest.fn(),
        startCamera: jest.fn(),
        enrollFromCanvas: jest.fn(),
        identifyFromCanvas: jest.fn(),
        captureVideoFrame: jest.fn(() => createCanvasStub(10, 10)),
        updateStatus: jest.fn(),
      };
    };

    await bootstrapFaceRecognitionApp({ documentRef: stubDoc, autoInit: false });
    expect(capturedConfig.normalizeMean).toBeUndefined();
    expect(capturedConfig.normalizeScale).toBeUndefined();
  });

  test('bootstrap ignores invalid document URL gracefully', async () => {
    const stubDoc = {
      getElementById: document.getElementById.bind(document),
      location: { href: 'not a url' },
    };

    global.__createFakeApp = () => ({
      initialize: jest.fn(),
      startCamera: jest.fn(),
      enrollFromCanvas: jest.fn(),
      identifyFromCanvas: jest.fn(),
      captureVideoFrame: jest.fn(() => createCanvasStub(10, 10)),
      updateStatus: jest.fn(),
    });

    await expect(bootstrapFaceRecognitionApp({ documentRef: stubDoc, autoInit: false })).resolves.toBeDefined();
  });

  test('bootstrap handles missing location on documentRef', async () => {
    const stubDoc = {
      getElementById: document.getElementById.bind(document),
      // no location property
    };

    global.__createFakeApp = () => ({
      initialize: jest.fn(),
      startCamera: jest.fn(),
      enrollFromCanvas: jest.fn(),
      identifyFromCanvas: jest.fn(),
      captureVideoFrame: jest.fn(() => createCanvasStub(10, 10)),
      updateStatus: jest.fn(),
    });

    await expect(bootstrapFaceRecognitionApp({ documentRef: stubDoc, autoInit: false })).resolves.toBeDefined();
  });
});
