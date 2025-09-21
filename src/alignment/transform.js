import { ARC_FACE_TEMPLATE } from '../constants/arcfaceTemplate.js';

function meanPoint(points) {
  const total = points.reduce(
    (acc, [x, y]) => {
      acc[0] += x;
      acc[1] += y;
      return acc;
    },
    [0, 0],
  );
  return [total[0] / points.length, total[1] / points.length];
}

function estimateSimilarityTransform(source, target = ARC_FACE_TEMPLATE) {
  if (!Array.isArray(source) || !Array.isArray(target)) {
    throw new TypeError('source and target must be arrays of points');
  }
  if (source.length !== target.length) {
    throw new Error('source and target must contain the same number of points');
  }
  if (source.length < 2) {
    throw new Error('At least two points are required to estimate a similarity transform');
  }

  const srcMean = meanPoint(source);
  const dstMean = meanPoint(target);

  let ss = 0;
  let sx = 0;
  let sy = 0;

  for (let i = 0; i < source.length; i += 1) {
    const srcX = source[i][0] - srcMean[0];
    const srcY = source[i][1] - srcMean[1];
    const dstX = target[i][0] - dstMean[0];
    const dstY = target[i][1] - dstMean[1];

    ss += srcX * srcX + srcY * srcY;
    sx += srcX * dstX + srcY * dstY;
    sy += srcX * dstY - srcY * dstX;
  }

  if (ss === 0) {
    throw new Error('Degenerate configuration: source points are collinear or identical');
  }

  const norm = Math.hypot(sx, sy);
  if (norm === 0) {
    throw new Error('Unable to compute rotation: covariance is zero');
  }

  const scale = norm / ss;
  const cos = sx / norm;
  const sin = sy / norm;

  const r00 = scale * cos;
  const r01 = -scale * sin;
  const r10 = scale * sin;
  const r11 = scale * cos;

  const tx = dstMean[0] - (r00 * srcMean[0] + r01 * srcMean[1]);
  const ty = dstMean[1] - (r10 * srcMean[0] + r11 * srcMean[1]);

  return [
    [r00, r01, tx],
    [r10, r11, ty],
  ];
}

function transformPoint(point, matrix) {
  const [x, y] = point;
  const [[a, b, tx], [c, d, ty]] = matrix;
  return [a * x + b * y + tx, c * x + d * y + ty];
}

function applySimilarityTransform(points, matrix) {
  return points.map((pt) => transformPoint(pt, matrix));
}

export { estimateSimilarityTransform, transformPoint, applySimilarityTransform };
