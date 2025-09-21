function dot(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function l2Norm(vector) {
  return Math.sqrt(vector.reduce((acc, value) => acc + value * value, 0));
}

function l2Normalize(vector) {
  const norm = l2Norm(vector);
  if (norm === 0) {
    throw new Error('Cannot normalize a zero vector');
  }
  return vector.map((value) => value / norm);
}

function cosineSimilarity(a, b) {
  return dot(a, b) / (l2Norm(a) * l2Norm(b));
}

export { dot, l2Norm, l2Normalize, cosineSimilarity };
