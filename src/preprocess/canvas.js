import { l2Normalize } from '../math/vector.js';

function canvasToCHWFloat32(canvas, { mean = 127.5, scale = 1 / 128 } = {}) {
  if (!canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('A canvas element is required');
  }
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { width, height } = canvas;
  const { data } = ctx.getImageData(0, 0, width, height);
  const size = width * height;
  const tensor = new Float32Array(3 * size);
  for (let i = 0, p = 0; i < size; i += 1, p += 4) {
    const r = (data[p] - mean) * scale;
    const g = (data[p + 1] - mean) * scale;
    const b = (data[p + 2] - mean) * scale;
    tensor[i] = r;
    tensor[i + size] = g;
    tensor[i + 2 * size] = b;
  }
  return tensor;
}

function normalizeEmbedding(vectorLike) {
  const vector = Array.from(vectorLike);
  return l2Normalize(vector);
}

export { canvasToCHWFloat32, normalizeEmbedding };
