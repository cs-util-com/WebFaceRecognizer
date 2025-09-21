import { mapBoxToOriginal, mapKeypointsToOriginal } from './letterbox.js';

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function decodeScrfdOutputs(outputs, {
  inputSize = 640,
  scoreThreshold = 0.4,
  strides = [8, 16, 32],
  applySigmoid = true,
} = {}) {
  const detections = [];
  strides.forEach((stride) => {
    const entry = outputs[stride];
    if (!entry) {
      return;
    }
    const { scores, boxes, keypoints } = entry;
    if (!scores || !boxes || !keypoints) {
      return;
    }
    const gridSize = Math.round(inputSize / stride);
    for (let idx = 0; idx < scores.length; idx += 1) {
      const rawScore = scores[idx];
      const score = applySigmoid ? sigmoid(rawScore) : rawScore;
      if (score < scoreThreshold) {
        continue;
      }
      const x = idx % gridSize;
      const y = Math.floor(idx / gridSize);
      const anchorX = (x + 0.5) * stride;
      const anchorY = (y + 0.5) * stride;
      const boxOffset = idx * 4;
      const left = boxes[boxOffset] * stride;
      const top = boxes[boxOffset + 1] * stride;
      const right = boxes[boxOffset + 2] * stride;
      const bottom = boxes[boxOffset + 3] * stride;
      const bbox = [
        anchorX - left,
        anchorY - top,
        anchorX + right,
        anchorY + bottom,
      ];
      const kp = [];
      const kpOffset = idx * 10;
      for (let point = 0; point < 5; point += 1) {
        const px = keypoints[kpOffset + point * 2] * stride + anchorX;
        const py = keypoints[kpOffset + point * 2 + 1] * stride + anchorY;
        kp.push([px, py]);
      }
      detections.push({ bbox, keypoints: kp, score, stride });
    }
  });
  return detections;
}

function intersectionOverUnion(boxA, boxB) {
  const x1 = Math.max(boxA[0], boxB[0]);
  const y1 = Math.max(boxA[1], boxB[1]);
  const x2 = Math.min(boxA[2], boxB[2]);
  const y2 = Math.min(boxA[3], boxB[3]);
  const interWidth = Math.max(0, x2 - x1);
  const interHeight = Math.max(0, y2 - y1);
  const interArea = interWidth * interHeight;
  const areaA = Math.max(0, boxA[2] - boxA[0]) * Math.max(0, boxA[3] - boxA[1]);
  const areaB = Math.max(0, boxB[2] - boxB[0]) * Math.max(0, boxB[3] - boxB[1]);
  const union = areaA + areaB - interArea;
  if (union <= 0) {
    return 0;
  }
  return interArea / union;
}

function nonMaxSuppression(detections, iouThreshold = 0.4) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept = [];
  while (sorted.length > 0) {
    const candidate = sorted.shift();
    let discard = false;
    for (let i = 0; i < kept.length; i += 1) {
      if (intersectionOverUnion(candidate.bbox, kept[i].bbox) > iouThreshold) {
        discard = true;
        break;
      }
    }
    if (!discard) {
      kept.push(candidate);
    }
  }
  return kept;
}

function postprocessDetections(detections, letterboxMeta) {
  if (!letterboxMeta) {
    return detections;
  }
  return detections.map((det) => ({
    score: det.score,
    stride: det.stride,
    bbox: mapBoxToOriginal(det.bbox, letterboxMeta),
    keypoints: mapKeypointsToOriginal(det.keypoints, letterboxMeta),
  }));
}

function normalizeScrfdOutput(rawOutputs, strides = [8, 16, 32]) {
  const normalized = {};
  const candidates = (base, stride) => [
    `${base}_${stride}`,
    `${base}${stride}`,
    `${base}_${stride}:0`,
    `${base}${stride}:0`,
  ];
  const find = (base, stride) => {
    const keys = candidates(base, stride);
    for (let i = 0; i < keys.length; i += 1) {
      if (rawOutputs[keys[i]]) {
        return rawOutputs[keys[i]].data || rawOutputs[keys[i]];
      }
    }
    const nested = rawOutputs[stride];
    if (nested && nested[base]) {
      return nested[base].data || nested[base];
    }
    return undefined;
  };

  strides.forEach((stride) => {
    const scores = find('scores', stride);
    const boxes = find('boxes', stride) || find('bbox', stride);
    const keypoints = find('kps', stride) || find('keypoints', stride);
    if (scores && boxes && keypoints) {
      normalized[stride] = { scores, boxes, keypoints };
    }
  });
  return normalized;
}

export {
  decodeScrfdOutputs,
  nonMaxSuppression,
  intersectionOverUnion,
  postprocessDetections,
  normalizeScrfdOutput,
};
