function ensureScores(scores) {
  if (!Array.isArray(scores) || scores.length === 0) {
    throw new Error('scores must be a non-empty array');
  }
  scores.forEach((value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      throw new Error('scores must contain finite numbers');
    }
  });
  return scores;
}

function normalizeSample(value, min, max, clamp) {
  if (!clamp) {
    return value;
  }
  if (value < min) {
    return min;
  }
  if (value > max) {
    return max;
  }
  return value;
}

function prepareHistogramOptions(bins, range) {
  if (!Number.isInteger(bins) || bins <= 0) {
    throw new Error('bins must be a positive integer');
  }
  if (!Array.isArray(range) || range.length !== 2 || range[0] >= range[1]) {
    throw new Error('range must contain [min, max] with min < max');
  }
  const [min, max] = range;
  const width = (max - min) / bins;
  if (width === 0) {
    throw new Error('range width must be positive');
  }
  return { min, max, width };
}

function computeHistogram(scores, { bins = 10, range = [-1, 1], clamp = true } = {}) {
  const samples = ensureScores(scores);
  const { min, max, width } = prepareHistogramOptions(bins, range);
  const histogram = new Array(bins).fill(0);
  samples.forEach((value) => {
    const sample = normalizeSample(value, min, max, clamp);
    if (sample < min || sample > max) {
      return;
    }
    let index = Math.floor((sample - min) / width);
    if (index >= bins) {
      index = bins - 1;
    }
    histogram[index] += 1;
  });
  return histogram;
}

function cumulativeDistribution(histogram) {
  if (!Array.isArray(histogram) || histogram.length === 0) {
    throw new Error('histogram must be a non-empty array');
  }
  histogram.forEach((value) => {
    if (!Number.isInteger(value) || value < 0) {
      throw new Error('histogram must contain non-negative integers');
    }
  });
  const total = histogram.reduce((acc, value) => acc + value, 0);
  if (total === 0) {
    return new Array(histogram.length).fill(0);
  }
  const result = [];
  let cumulative = 0;
  histogram.forEach((count) => {
    cumulative += count;
    result.push(cumulative / total);
  });
  return result;
}

function normalizeHistogram(histogram, { smooth = 0 } = {}) {
  if (smooth < 0) {
    throw new Error('smooth must be non-negative');
  }
  const base = cumulativeDistribution(histogram).map((value, index) => histogram[index]);
  const smoothed = base.map((count) => count + smooth);
  const total = smoothed.reduce((acc, value) => acc + value, 0);
  if (total === 0) {
    return new Array(smoothed.length).fill(0);
  }
  return smoothed.map((value) => value / total);
}

function mergeHistograms(...histograms) {
  if (histograms.length === 0) {
    throw new Error('at least one histogram is required');
  }
  const length = histograms[0].length;
  histograms.forEach((histogram) => {
    if (!Array.isArray(histogram) || histogram.length !== length) {
      throw new Error('all histograms must have the same length');
    }
    histogram.forEach((value) => {
      if (!Number.isInteger(value) || value < 0) {
        throw new Error('histograms must contain non-negative integers');
      }
    });
  });
  return histograms[0].map((_, index) => histograms.reduce((sum, histogram) => sum + histogram[index], 0));
}

export { computeHistogram, cumulativeDistribution, normalizeHistogram, mergeHistograms };
