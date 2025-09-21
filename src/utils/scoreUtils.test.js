import { classifySimilarity, blendScores } from './scoreUtils.js';

describe('score utilities', () => {
  describe('classifySimilarity', () => {
    test('classifies strong, possible, weak and invalid scores', () => {
      expect(classifySimilarity(0.95)).toBe('strong');
      expect(classifySimilarity(0.5)).toBe('possible');
      expect(classifySimilarity(-0.2)).toBe('weak');
      expect(classifySimilarity(-2)).toBe('invalid');
    });

    test('rejects non-numeric scores', () => {
      expect(() => classifySimilarity(NaN)).toThrow('finite number');
      expect(() => classifySimilarity('bad')).toThrow('finite number');
    });
  });

  describe('blendScores', () => {
    test('blends scores using the provided weight', () => {
      expect(blendScores(0.8, 0.4, 0.75)).toBeCloseTo(0.7);
      expect(blendScores(0.6, 0.2)).toBeCloseTo(0.4);
      expect(blendScores(0.3, 0.9, 0)).toBeCloseTo(0.9);
      expect(blendScores(0.3, 0.9, 1)).toBeCloseTo(0.3);
    });

    test('validates scores and weights', () => {
      expect(() => blendScores('bad', 0.2)).toThrow('scores must be numbers');
      expect(() => blendScores(0.3, 0.2, -0.1)).toThrow('within [0,1]');
      expect(() => blendScores(0.3, 0.2, 1.1)).toThrow('within [0,1]');
      expect(() => blendScores(0.3, 0.2, NaN)).toThrow('weight must be a number');
    });
  });
});
