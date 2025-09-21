function ensureArray(values, name) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error(`${name} must be a non-empty array`);
  }
  return values;
}

function analyzeScores(scores) {
  const samples = ensureArray(scores, 'scores');
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  samples.forEach((value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      throw new Error('scores must contain finite numbers');
    }
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
  });
  return { min, max, mean: sum / samples.length };
}

function estimateThreshold(positives, negatives, targetFAR = 1e-3) {
  const pos = ensureArray(positives, 'positives');
  const neg = ensureArray(negatives, 'negatives');
  if (targetFAR <= 0 || targetFAR >= 1) {
    throw new Error('targetFAR must be between 0 and 1');
  }
  const sortedNeg = [...neg].sort((a, b) => a - b);
  const index = Math.min(sortedNeg.length - 1, Math.max(0, Math.ceil((1 - targetFAR) * sortedNeg.length) - 1));
  const candidate = sortedNeg[index];
  const positiveFloor = Math.min(...pos);
  let threshold = Math.max(candidate, positiveFloor);
  threshold = Math.max(-1, Math.min(1, threshold));
  const falseAccepts = sortedNeg.filter((value) => value >= threshold).length;
  const far = falseAccepts / sortedNeg.length;
  return { threshold, far };
}

export { analyzeScores, estimateThreshold };
