import { ARC_FACE_TEMPLATE } from '../constants/arcfaceTemplate.js';
import { calculateLetterbox } from '../detection/letterbox.js';
import {
  decodeScrfdOutputs,
  nonMaxSuppression,
  postprocessDetections,
  normalizeScrfdOutput,
} from '../detection/scrfdDecoder.js';
import { estimateSimilarityTransform } from '../alignment/transform.js';
import { canvasToCHWFloat32, canvasToCHWUint8, normalizeEmbedding } from '../preprocess/canvas.js';
import { FaceEmbeddingStore } from '../store/faceStore.js';
import { loadOrtRuntime } from '../runtime/loadOrt.js';

function createCanvasFactory(documentRef) {
  return (width, height) => {
    if (!documentRef) {
      throw new Error('No document available to create a canvas element');
    }
    const canvas = documentRef.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  };
}

class FaceRecognitionApp {
  constructor(options = {}) {
    const defaults = {
      documentRef: typeof document !== 'undefined' ? document : undefined,
      videoElement: null,
      overlayCanvas: null,
      statusElement: null,
      runtimeLoader: loadOrtRuntime,
      store: new FaceEmbeddingStore(),
  detectorModelUrl: '/models/scrfd_2.5g_bnkps.onnx',
  // Default to INT8 ArcFace model; can be overridden via query params or config
  embedderModelUrl: '/models/arcfaceresnet100-11-int8.onnx',
      detectorInputName: 'input',
  embedderInputName: 'data',
  embedderOutputName: 'fc1',
  // For INT8 model variants that accept uint8 input, allow choosing tensor type
  embedderInputType: 'float32', // 'float32' | 'uint8'
  // Normalization parameters for float32 preprocessing
  normalizeMean: 127.5,
  normalizeScale: 1 / 128,
      detectorInputSize: 640,
      alignedSize: 112,
      detectorStrides: [8, 16, 32],
      detectionScoreThreshold: 0.45,
      nmsThreshold: 0.4,
      arcfaceTemplate: ARC_FACE_TEMPLATE,
    };
    const config = { ...defaults, ...options };
    const {
      documentRef,
      videoElement,
      overlayCanvas,
      statusElement,
      runtimeLoader,
      store,
      detectorModelUrl,
      embedderModelUrl,
      detectorInputName,
  embedderInputName,
      embedderOutputName,
  embedderInputType,
  normalizeMean,
  normalizeScale,
      detectorInputSize,
      alignedSize,
      detectorStrides,
      detectionScoreThreshold,
      nmsThreshold,
      arcfaceTemplate,
      createCanvas,
    } = config;

    this.document = documentRef;
    this.videoElement = videoElement;
    this.overlayCanvas = overlayCanvas;
    this.statusElement = statusElement;
    this.runtimeLoader = runtimeLoader;
    this.store = store;
    this.detectorModelUrl = detectorModelUrl;
    this.embedderModelUrl = embedderModelUrl;
    this.detectorInputName = detectorInputName;
    this.embedderInputName = embedderInputName;
  this.embedderOutputName = embedderOutputName;
  this.embedderInputType = embedderInputType;
  this.normalizeMean = normalizeMean;
  this.normalizeScale = normalizeScale;
    this.detectorInputSize = detectorInputSize;
    this.alignedSize = alignedSize;
    this.detectorStrides = detectorStrides;
    this.detectionScoreThreshold = detectionScoreThreshold;
    this.nmsThreshold = nmsThreshold;
    this.arcfaceTemplate = arcfaceTemplate;
    this.createCanvas = createCanvas || createCanvasFactory(this.document);
    this.captureCanvas = this.createCanvas(detectorInputSize, detectorInputSize);
    this.captureContext = this.captureCanvas.getContext('2d');
  }

  updateStatus(message) {
    if (this.statusElement) {
      this.statusElement.textContent = message;
    }
    if (typeof console !== 'undefined' && console.debug) {
      console.debug('[FaceRecognitionApp] status:', message);
    }
  }

  async initialize() {
    this.updateStatus('Loading ONNX Runtime...');
    if (typeof console !== 'undefined' && console.time) {
      console.time('[FaceRecognitionApp] initialize');
      console.time('[FaceRecognitionApp] loadOrt');
    }
    const { ort, executionProviders } = await this.runtimeLoader();
    if (typeof console !== 'undefined' && console.timeEnd) {
      console.timeEnd('[FaceRecognitionApp] loadOrt');
    }
    this.ort = ort;
    this.executionProviders = executionProviders;
    if (typeof console !== 'undefined' && console.info) {
      console.info('[FaceRecognitionApp] ORT ready with providers:', executionProviders);
    }
    await this.loadModelSessions();
    this.updateStatus(`Loaded models (${executionProviders.join(', ')})`);
    if (typeof console !== 'undefined' && console.timeEnd) {
      console.timeEnd('[FaceRecognitionApp] initialize');
    }
  }

  async loadModelSessions() {
    if (!this.ort) {
      throw new Error('ONNX Runtime not loaded');
    }
    const sessionOptions = { executionProviders: this.executionProviders };
    if (typeof console !== 'undefined' && console.debug) {
      console.debug('[FaceRecognitionApp] Creating sessions with options:', sessionOptions);
      console.debug('[FaceRecognitionApp] Detector model URL:', this.detectorModelUrl);
      console.debug('[FaceRecognitionApp] Embedder model URL:', this.embedderModelUrl);
    }
    if (typeof console !== 'undefined' && console.time) {
      console.time('[FaceRecognitionApp] create detector session');
    }
    try {
      this.detectorSession = await this.ort.InferenceSession.create(this.detectorModelUrl, sessionOptions);
    } catch (e) {
      const hint = `Failed to load detector model at ${this.detectorModelUrl}. Ensure the file exists and is served from the same origin (e.g., place it under /models).`;
      throw new Error(e && e.message ? `${e.message}`.includes('failed to load external data file') ? `${hint}` : `${hint}\n${e.message}` : hint);
    }
    if (typeof console !== 'undefined' && console.timeEnd) {
      console.timeEnd('[FaceRecognitionApp] create detector session');
    }
    if (typeof console !== 'undefined' && console.time) {
      console.time('[FaceRecognitionApp] create embedder session');
    }
    try {
      this.embedderSession = await this.ort.InferenceSession.create(this.embedderModelUrl, sessionOptions);
    } catch (e) {
      const hint = `Failed to load embedder model at ${this.embedderModelUrl}. Ensure the file exists and is served from the same origin (e.g., place it under /models).`;
      throw new Error(e && e.message ? `${hint}\n${e.message}` : hint);
    }
    if (typeof console !== 'undefined' && console.timeEnd) {
      console.timeEnd('[FaceRecognitionApp] create embedder session');
    }
  }

  async startCamera(constraints = { video: { width: 1280, height: 720 } }) {
    if (!this.videoElement) {
      throw new Error('Video element is not configured');
    }
    if (typeof console !== 'undefined' && console.debug) {
      console.debug('[FaceRecognitionApp] Starting camera with constraints:', constraints);
    }
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.videoElement.srcObject = stream;
    await this.videoElement.play();
    return stream;
  }

  captureVideoFrame() {
    if (!this.videoElement) {
      throw new Error('Video element is not configured');
    }
    const { videoWidth, videoHeight } = this.videoElement;
    this.captureCanvas.width = videoWidth;
    this.captureCanvas.height = videoHeight;
    this.captureContext.drawImage(this.videoElement, 0, 0, videoWidth, videoHeight);
    return this.captureCanvas;
  }

  prepareDetectorInput(sourceCanvas) {
    const letterboxMeta = calculateLetterbox(sourceCanvas.width, sourceCanvas.height, this.detectorInputSize);
    const detectorCanvas = this.createCanvas(this.detectorInputSize, this.detectorInputSize);
    const ctx = detectorCanvas.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, this.detectorInputSize, this.detectorInputSize);
    ctx.drawImage(
      sourceCanvas,
      0,
      0,
      sourceCanvas.width,
      sourceCanvas.height,
      letterboxMeta.padX,
      letterboxMeta.padY,
      letterboxMeta.scaledWidth,
      letterboxMeta.scaledHeight,
    );
    const tensor = canvasToCHWFloat32(detectorCanvas);
    const input = new this.ort.Tensor('float32', tensor, [1, 3, this.detectorInputSize, this.detectorInputSize]);
    return { input, letterboxMeta };
  }

  async detectFacesFromCanvas(canvas) {
    const { input, letterboxMeta } = this.prepareDetectorInput(canvas);
    const feeds = { [this.detectorInputName]: input };
    const outputMap = await this.detectorSession.run(feeds);
    const normalized = normalizeScrfdOutput(outputMap, this.detectorStrides);
    const decoded = decodeScrfdOutputs(normalized, {
      inputSize: this.detectorInputSize,
      scoreThreshold: this.detectionScoreThreshold,
      strides: this.detectorStrides,
    });
    const filtered = nonMaxSuppression(decoded, this.nmsThreshold);
    return postprocessDetections(filtered, letterboxMeta);
  }

  alignFace(canvas, keypoints) {
    const matrix = estimateSimilarityTransform(keypoints, this.arcfaceTemplate);
    const aligned = this.createCanvas(this.alignedSize, this.alignedSize);
    const ctx = aligned.getContext('2d');
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, this.alignedSize, this.alignedSize);
    ctx.setTransform(
      matrix[0][0],
      matrix[1][0],
      matrix[0][1],
      matrix[1][1],
      matrix[0][2],
      matrix[1][2],
    );
    ctx.drawImage(canvas, 0, 0);
    return { canvas: aligned, matrix };
  }

  async embedAlignedCanvas(alignedCanvas) {
    let input;
    if (this.embedderInputType === 'uint8') {
      const tensor = canvasToCHWUint8(alignedCanvas);
      input = new this.ort.Tensor('uint8', tensor, [1, 3, this.alignedSize, this.alignedSize]);
    } else {
      const tensor = canvasToCHWFloat32(alignedCanvas, { mean: this.normalizeMean, scale: this.normalizeScale });
      input = new this.ort.Tensor('float32', tensor, [1, 3, this.alignedSize, this.alignedSize]);
    }
    const feeds = { [this.embedderInputName]: input };
    const output = await this.embedderSession.run(feeds);
    const rawEmbedding = output[this.embedderOutputName];
    const data = rawEmbedding.data || rawEmbedding;
    return normalizeEmbedding(data);
  }

  async identifyFromCanvas(canvas) {
    const detections = await this.detectFacesFromCanvas(canvas);
    const results = [];
    for (let i = 0; i < detections.length; i += 1) {
      const { keypoints } = detections[i];
      const { canvas: aligned } = this.alignFace(canvas, keypoints);
      const embedding = await this.embedAlignedCanvas(aligned);
      const match = this.store.match(embedding);
      results.push({ detection: detections[i], match });
    }
    return results;
  }

  async enrollFromCanvas(canvas, id, metadata = {}) {
    const detections = await this.detectFacesFromCanvas(canvas);
    if (detections.length === 0) {
      throw new Error('No face detected for enrollment');
    }
    if (typeof console !== 'undefined' && console.debug) {
      console.debug('[FaceRecognitionApp] Enrolling id:', id, 'metadata:', metadata);
    }
    const { keypoints } = detections[0];
    const { canvas: aligned } = this.alignFace(canvas, keypoints);
    const embedding = await this.embedAlignedCanvas(aligned);
    return this.store.enroll(id, embedding, metadata);
  }
}

export { FaceRecognitionApp };
