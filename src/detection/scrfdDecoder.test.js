const {
  decodeScrfdOutputs,
  nonMaxSuppression,
  intersectionOverUnion,
  postprocessDetections,
  normalizeScrfdOutput,
} = require('./scrfdDecoder.js');

function createOutputForStride(stride, inputSize) {
  const grid = inputSize / stride;
  const anchors = grid * grid;
  const scores = new Float32Array(anchors).fill(-10);
  const boxes = new Float32Array(anchors * 4).fill(0);
  const keypoints = new Float32Array(anchors * 10).fill(0);
  const activeIndex = 1;
  scores[activeIndex] = 10;
  boxes.set([0.4, 0.4, 0.4, 0.4], activeIndex * 4);
  for (let i = 0; i < 5; i += 1) {
    keypoints[activeIndex * 10 + i * 2] = (i - 2) * 0.01;
    keypoints[activeIndex * 10 + i * 2 + 1] = (i - 2) * 0.01;
  }
  return { scores, boxes, keypoints };
}

describe('SCRFD decoder', () => {
  test('decodes detections and applies NMS', () => {
    const stride = 8;
    const inputSize = 32;
    const outputs = { [stride]: createOutputForStride(stride, inputSize) };
    const detections = decodeScrfdOutputs(outputs, { strides: [stride], inputSize, scoreThreshold: 0.1 });
    expect(detections).toHaveLength(1);
    const det = detections[0];
    expect(det.bbox[0]).toBeLessThan(det.bbox[2]);
    expect(det.keypoints).toHaveLength(5);
    const suppressed = nonMaxSuppression([...detections, ...detections], 0.3);
    expect(suppressed).toHaveLength(1);
    const meta = { scale: 0.5, padX: 4, padY: 6 };
    const processed = postprocessDetections(suppressed, meta);
    expect(processed[0].bbox[0]).toBeCloseTo((det.bbox[0] - 4) / 0.5);
    const unchanged = postprocessDetections(suppressed);
    expect(unchanged[0].bbox).toEqual(suppressed[0].bbox);
  });

  test('normalizes output maps from common naming schemes', () => {
    const scores = { data: new Float32Array([0]) };
    const boxes = { data: new Float32Array([0, 0, 0, 0]) };
    const kps = { data: new Float32Array(10) };
    const normalized = normalizeScrfdOutput({ scores_8: scores, bbox_8: boxes, kps_8: kps }, [8]);
    expect(normalized[8].scores).toBe(scores.data);
    const normalizedAlt = normalizeScrfdOutput({ 'scores8:0': scores, 'boxes8:0': boxes, 'keypoints8:0': kps }, [8]);
    expect(normalizedAlt[8].keypoints).toBe(kps.data);
    const normalizedBare = normalizeScrfdOutput({ scores8: new Float32Array([0]), boxes8: new Float32Array([0, 0, 0, 0]), kps8: new Float32Array(10) }, [8]);
    expect(normalizedBare[8].boxes).toBeInstanceOf(Float32Array);
  });

  test('computes IOU', () => {
    const iou = intersectionOverUnion([0, 0, 10, 10], [5, 5, 15, 15]);
    expect(iou).toBeCloseTo(25 / (100 + 100 - 25));
    const zero = intersectionOverUnion([0, 0, 0, 0], [0, 0, 0, 0]);
    expect(zero).toBe(0);
  });

  test('handles missing stride outputs gracefully', () => {
    const decoded = decodeScrfdOutputs({}, { strides: [8] });
    expect(decoded).toHaveLength(0);
    const suppressed = nonMaxSuppression([], 0.5);
    expect(suppressed).toEqual([]);
    const missing = decodeScrfdOutputs({ 8: { scores: new Float32Array(1) } }, { strides: [8] });
    expect(missing).toHaveLength(0);
    const nested = normalizeScrfdOutput({
      8: {
        scores: { data: new Float32Array([0]) },
        boxes: { data: new Float32Array([0, 0, 0, 0]) },
        kps: { data: new Float32Array(10) },
      },
    }, [8]);
    expect(nested[8].scores).toBeInstanceOf(Float32Array);
  });

  test('supports disabling sigmoid application', () => {
    const stride = 8;
    const outputs = {
      [stride]: {
        scores: new Float32Array([0, 0.9]),
        boxes: new Float32Array([0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1]),
        keypoints: new Float32Array(20),
      },
    };
    const detections = decodeScrfdOutputs(outputs, { strides: [stride], inputSize: 16, scoreThreshold: 0.5, applySigmoid: false });
    expect(detections).toHaveLength(1);
    expect(detections[0].score).toBeCloseTo(0.9);
  });

  test('uses default decoding options when none are provided', () => {
    const outputs = {
      8: {
        scores: new Float32Array([-10, 10]),
        boxes: new Float32Array([0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2]),
        keypoints: new Float32Array(20),
      },
    };
    const detections = decodeScrfdOutputs(outputs);
    expect(detections).toHaveLength(1);
    expect(detections[0].stride).toBe(8);
  });

  test('nonMaxSuppression retains non-overlapping candidates', () => {
    const detections = [
      { bbox: [0, 0, 10, 10], score: 0.9 },
      { bbox: [100, 100, 110, 110], score: 0.8 },
    ];
    const kept = nonMaxSuppression(detections, 0.5);
    expect(kept).toHaveLength(2);
  });

  test('normalizeScrfdOutput skips incomplete stride entries', () => {
    const normalized = normalizeScrfdOutput({
      8: {
        scores: { data: new Float32Array([0]) },
      },
      boxes_16: { data: new Float32Array(4) },
      kps_16: { data: new Float32Array(10) },
      32: {
        other: {},
      },
    }, [8, 16, 32]);
    expect(normalized[8]).toBeUndefined();
    expect(normalized[16]).toBeUndefined();
    expect(normalized[32]).toBeUndefined();
  });
});
