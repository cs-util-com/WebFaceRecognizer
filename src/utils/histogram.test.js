import { computeHistogram, cumulativeDistribution, normalizeHistogram, mergeHistograms } from './histogram.js';

describe('histogram utilities', () => {
  test('computes histogram with clamping', () => {
    const histogram = computeHistogram([-1.5, -0.5, 0, 0.5, 1.2], { bins: 4 });
    expect(histogram.reduce((acc, value) => acc + value, 0)).toBe(5);
  });

  test('computes histogram without clamping', () => {
    const histogram = computeHistogram([0, 1, 2], { bins: 2, range: [0, 1], clamp: false });
    expect(histogram).toEqual([1, 1]);
  });

  test('validates histogram parameters', () => {
    expect(() => computeHistogram([], {})).toThrow('non-empty');
    expect(() => computeHistogram([NaN], {})).toThrow('finite numbers');
    expect(() => computeHistogram([0], { bins: 0 })).toThrow('positive integer');
    expect(() => computeHistogram([0], { range: [1, 0] })).toThrow('range');
  });

  test('skips out-of-range values when not clamping', () => {
    const histogram = computeHistogram([-2, 0, 2], { bins: 2, range: [-1, 1], clamp: false });
    expect(histogram).toEqual([0, 1]);
  });

  test('computes cumulative distribution', () => {
    const histogram = [1, 2, 3];
    const cdf = cumulativeDistribution(histogram);
    expect(cdf).toEqual([1 / 6, 3 / 6, 1]);
  });

  test('handles zero-filled histograms', () => {
    expect(cumulativeDistribution([0, 0, 0])).toEqual([0, 0, 0]);
  });

  test('validates cumulative inputs', () => {
    expect(() => cumulativeDistribution([])).toThrow('non-empty');
    expect(() => cumulativeDistribution([1, -1])).toThrow('non-negative');
  });

  test('normalizes histogram with smoothing', () => {
    const normalized = normalizeHistogram([1, 1], { smooth: 1 });
    expect(normalized[0]).toBeCloseTo(0.5);
  });

  test('normalizes histogram with zero total', () => {
    const normalized = normalizeHistogram([0, 0]);
    expect(normalized).toEqual([0, 0]);
  });

  test('validates normalization inputs', () => {
    expect(() => normalizeHistogram([1, 1], { smooth: -1 })).toThrow('smooth');
  });

  test('merges histograms', () => {
    const merged = mergeHistograms([1, 2], [3, 4]);
    expect(merged).toEqual([4, 6]);
  });

  test('validates merge inputs', () => {
    expect(() => mergeHistograms()).toThrow('at least one');
    expect(() => mergeHistograms([1], [1, 2])).toThrow('same length');
    expect(() => mergeHistograms([1], [-1])).toThrow('non-negative');
  });
});
