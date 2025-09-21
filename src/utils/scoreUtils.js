function classifySimilarity(score) {
  if (typeof score !== 'number' || Number.isNaN(score)) {
    throw new Error('score must be a finite number');
  }
  if (score >= 0.8) {
    return 'strong';
  }
  if (score >= 0.5) {
    return 'possible';
  }
  if (score >= -1) {
    return 'weak';
  }
  return 'invalid';
}

function blendScores(primary, secondary, weight = 0.5) {
  if (typeof primary !== 'number' || typeof secondary !== 'number') {
    throw new Error('scores must be numbers');
  }
  if (typeof weight !== 'number' || Number.isNaN(weight)) {
    throw new Error('weight must be a number');
  }
  if (weight < 0 || weight > 1) {
    throw new Error('weight must be within [0,1]');
  }
  return primary * weight + secondary * (1 - weight);
}

export { classifySimilarity, blendScores };
