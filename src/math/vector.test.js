const { dot, l2Norm, l2Normalize, cosineSimilarity } = require('./vector.js');

describe('vector utilities', () => {
  test('dot product and norm', () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(l2Norm([3, 4])).toBe(5);
  });

  test('normalizes and computes cosine similarity', () => {
    const normalized = l2Normalize([3, 4]);
    expect(normalized[0]).toBeCloseTo(0.6);
    expect(normalized[1]).toBeCloseTo(0.8);
    expect(() => l2Normalize([0, 0])).toThrow();
    const cosine = cosineSimilarity([1, 0, 0], [0.5, 0, 0]);
    expect(cosine).toBeCloseTo(1);
  });

  test('validates vector lengths', () => {
    expect(() => dot([1], [1, 2])).toThrow('same length');
  });
});
