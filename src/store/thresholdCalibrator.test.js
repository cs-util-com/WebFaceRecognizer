import { analyzeScores, estimateThreshold } from './thresholdCalibrator.js';

describe('threshold calibration utilities', () => {
  test('computes descriptive statistics', () => {
    const stats = analyzeScores([0.1, 0.2, 0.3]);
    expect(stats.min).toBeCloseTo(0.1);
    expect(stats.max).toBeCloseTo(0.3);
    expect(stats.mean).toBeCloseTo(0.2);
  });

  test('rejects invalid score arrays', () => {
    expect(() => analyzeScores([])).toThrow('scores');
    expect(() => analyzeScores([NaN])).toThrow('finite numbers');
  });

  test('estimates threshold with FAR constraint', () => {
    const { threshold, far } = estimateThreshold([0.8, 0.85, 0.9], [0.1, 0.2, 0.3, 0.4], 0.5);
    expect(threshold).toBeGreaterThan(0.4);
    expect(far).toBeLessThanOrEqual(0.5);
  });

  test('enforces input validation', () => {
    expect(() => estimateThreshold([], [0.1], 0.1)).toThrow('positives');
    expect(() => estimateThreshold([0.5], [], 0.1)).toThrow('negatives');
    expect(() => estimateThreshold([0.5], [0.1], 0)).toThrow('targetFAR');
    expect(() => estimateThreshold([0.5], [0.1], 1)).toThrow('targetFAR');
  });

  test('clamps threshold to score range', () => {
    const { threshold } = estimateThreshold([0.2], [0.9, 0.95], 0.1);
    expect(threshold).toBeLessThanOrEqual(1);
  });
});
