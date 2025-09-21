const { estimateSimilarityTransform, transformPoint, applySimilarityTransform } = require('./transform.js');
const { ARC_FACE_TEMPLATE } = require('../constants/arcfaceTemplate.js');

describe('estimateSimilarityTransform', () => {
  test('returns identity when source equals target', () => {
    const matrix = estimateSimilarityTransform(ARC_FACE_TEMPLATE, ARC_FACE_TEMPLATE);
    expect(matrix[0][0]).toBeCloseTo(1, 5);
    expect(matrix[0][1]).toBeCloseTo(0, 5);
    expect(matrix[1][0]).toBeCloseTo(0, 5);
    expect(matrix[1][1]).toBeCloseTo(1, 5);
    const point = transformPoint([10, 20], matrix);
    expect(point[0]).toBeCloseTo(10, 5);
    expect(point[1]).toBeCloseTo(20, 5);
  });

  test('estimates rotation and translation', () => {
    const source = [
      [0, 0],
      [10, 0],
      [5, 5],
    ];
    const angle = Math.PI / 4;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const target = source.map(([x, y]) => [cos * x - sin * y + 3, sin * x + cos * y - 2]);
    const matrix = estimateSimilarityTransform(source, target);
    const transformed = applySimilarityTransform(source, matrix);
    transformed.forEach((point, index) => {
      expect(point[0]).toBeCloseTo(target[index][0], 4);
      expect(point[1]).toBeCloseTo(target[index][1], 4);
    });
  });

  test('validates input arrays', () => {
    expect(() => estimateSimilarityTransform('bad', [])).toThrow(TypeError);
    expect(() => estimateSimilarityTransform([[0, 0]], [[0, 0]])).toThrow('At least two points');
    expect(() => estimateSimilarityTransform([[0, 0], [0, 0]], [[0, 0]])).toThrow('same number of points');
    expect(() => estimateSimilarityTransform([[0, 0], [0, 0]], [[0, 0], [0, 0]])).toThrow('Degenerate configuration');
    expect(() => estimateSimilarityTransform([[0, 0], [1, 0]], [[0, 0], [0, 0]])).toThrow('Unable to compute rotation');
  });

  test('uses the default ArcFace template when target is omitted', () => {
    const offsetSource = ARC_FACE_TEMPLATE.map(([x, y]) => [x + 1, y - 1]);
    const matrix = estimateSimilarityTransform(offsetSource);
    const transformed = applySimilarityTransform(offsetSource, matrix);
    transformed.forEach((point, index) => {
      expect(point[0]).toBeCloseTo(ARC_FACE_TEMPLATE[index][0], 4);
      expect(point[1]).toBeCloseTo(ARC_FACE_TEMPLATE[index][1], 4);
    });
  });
});
