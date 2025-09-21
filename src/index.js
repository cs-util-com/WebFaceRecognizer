import { FaceRecognitionApp } from './app/faceRecognitionApp.js';

function selectElement(documentRef, id) {
  const el = documentRef.getElementById(id);
  if (!el) {
    throw new Error(`Element with id "${id}" not found`);
  }
  return el;
}

function createDefaultAppConfig(documentRef = document) {
  const videoElement = selectElement(documentRef, 'camera');
  const overlayCanvas = selectElement(documentRef, 'overlay');
  const statusElement = selectElement(documentRef, 'status');
  return {
    documentRef,
    videoElement,
    overlayCanvas,
    statusElement,
  };
}

function drawDetections(canvas, detections, matches = []) {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  detections.forEach((det, index) => {
    const [x1, y1, x2, y2] = det.bbox;
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.9)';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const match = matches[index];
    if (match && match.match) {
      const label = `${match.match.id} (${match.match.score.toFixed(2)})`;
      ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
      ctx.fillRect(x1, Math.max(0, y1 - 20), ctx.measureText(label).width + 10, 20);
      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 + 5, Math.max(12, y1 - 5));
    }
    ctx.fillStyle = 'rgba(59, 130, 246, 0.6)';
    det.keypoints.forEach(([kx, ky]) => {
      ctx.beginPath();
      ctx.arc(kx, ky, 2, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

function renderMatchList(listElement, matches) {
  if (!listElement) {
    return;
  }
  const doc = listElement.ownerDocument || document;
  listElement.innerHTML = '';
  matches.forEach(({ detection, match }) => {
    const item = doc.createElement('li');
    if (match) {
      item.textContent = `${match.id} — score ${match.score.toFixed(3)}`;
    } else {
      item.textContent = `Unknown face at (${detection.bbox.map((v) => v.toFixed(0)).join(', ')})`;
    }
    listElement.appendChild(item);
  });
}

async function handleIdentify(app, overlayCanvas, resultsList) {
  const frame = app.captureVideoFrame();
  overlayCanvas.width = frame.width;
  overlayCanvas.height = frame.height;
  const matches = await app.identifyFromCanvas(frame);
  drawDetections(overlayCanvas, matches.map((m) => m.detection), matches);
  renderMatchList(resultsList, matches);
}

async function handleEnroll(app, overlayCanvas, resultsList, identityInput) {
  const identity = identityInput.value.trim();
  if (!identity) {
    app.updateStatus('Provide an identity label before enrolling.');
    return;
  }
  const frame = app.captureVideoFrame();
  overlayCanvas.width = frame.width;
  overlayCanvas.height = frame.height;
  try {
    await app.enrollFromCanvas(frame, identity, { enrolledAt: Date.now() });
    app.updateStatus(`Enrolled ${identity}`);
    identityInput.value = '';
    renderMatchList(resultsList, []);
    drawDetections(overlayCanvas, []);
  } catch (err) {
    app.updateStatus(err.message);
  }
}

async function bootstrapFaceRecognitionApp({ documentRef = document, autoInit = true } = {}) {
  const config = createDefaultAppConfig(documentRef);

  // Optional: allow overriding model URLs via query parameters
  // ?detector=/models/scrfd_2.5g_bnkps.onnx&embedder=/models/arcface_r100.onnx
  try {
    const href = (documentRef && documentRef.location && documentRef.location.href) || '';
    const params = new URL(href).searchParams;
    const detectorOverride = params.get('detector');
    const embedderOverride = params.get('embedder');
    const embedderInputType = params.get('embedderInputType'); // 'float32' | 'uint8'
    const embedderInputName = params.get('embedderInputName');
    const embedderOutputName = params.get('embedderOutputName');
    const normalizeMean = params.get('normalizeMean');
    const normalizeScale = params.get('normalizeScale');
    if (detectorOverride) config.detectorModelUrl = detectorOverride;
    if (embedderOverride) config.embedderModelUrl = embedderOverride;
    if (embedderInputType) config.embedderInputType = embedderInputType;
    if (embedderInputName) config.embedderInputName = embedderInputName;
    if (embedderOutputName) config.embedderOutputName = embedderOutputName;
    if (normalizeMean !== null) {
      const v = Number(normalizeMean);
      if (!Number.isNaN(v)) config.normalizeMean = v;
    }
    if (normalizeScale !== null) {
      const v = Number(normalizeScale);
      if (!Number.isNaN(v)) config.normalizeScale = v;
    }
  } catch {
    // ignore URL parsing errors in non-browser contexts
  }

  const app = new FaceRecognitionApp(config);
  if (autoInit) {
    await app.initialize();
  }
  const startButton = selectElement(documentRef, 'start');
  const enrollButton = selectElement(documentRef, 'enroll');
  const identifyButton = selectElement(documentRef, 'identify');
  const uploadInput = selectElement(documentRef, 'upload');
  const resultsList = selectElement(documentRef, 'results');
  const identityInput = selectElement(documentRef, 'identity');
  const overlayCanvas = config.overlayCanvas;

  startButton.addEventListener('click', async () => {
    try {
      await app.startCamera();
      app.updateStatus('Camera started');
    } catch (err) {
      app.updateStatus(err.message);
    }
  });

  enrollButton.addEventListener('click', async () => {
    try {
      await handleEnroll(app, overlayCanvas, resultsList, identityInput);
    } catch (err) {
      app.updateStatus(err.message);
    }
  });

  identifyButton.addEventListener('click', async () => {
    try {
      await handleIdentify(app, overlayCanvas, resultsList);
    } catch (err) {
      app.updateStatus(err.message);
    }
  });

  // Process a single uploaded image from disk
  uploadInput.addEventListener('change', async (e) => {
    const file = e.target && e.target.files && e.target.files[0];
    if (!file) return;
    try {
      // Decode image into an ImageBitmap for fast draw
      const img = await createImageBitmap(file);
      // Draw into a working canvas sized to the image
      const workCanvas = documentRef.createElement('canvas');
      workCanvas.width = img.width;
      workCanvas.height = img.height;
      const ctx = workCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      // Resize overlay to match and run identify on this static frame
      overlayCanvas.width = workCanvas.width;
      overlayCanvas.height = workCanvas.height;
      const matches = await app.identifyFromCanvas(workCanvas);
      drawDetections(overlayCanvas, matches.map((m) => m.detection), matches);
      renderMatchList(resultsList, matches);
      app.updateStatus(`Processed uploaded image (${workCanvas.width}×${workCanvas.height})`);
    } catch (err) {
      app.updateStatus(err && err.message ? err.message : String(err));
    } finally {
      // reset so selecting the same file again re-triggers change
      e.target.value = '';
    }
  });

  return app;
}

  if (typeof window !== 'undefined') {
    window.bootstrapFaceRecognitionApp = bootstrapFaceRecognitionApp;
  }

  export { bootstrapFaceRecognitionApp, createDefaultAppConfig, drawDetections, renderMatchList };
