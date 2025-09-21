const { ARC_FACE_TEMPLATE } = require('./arcfaceTemplate.js');

describe('ARC_FACE_TEMPLATE', () => {
  test('contains five landmark coordinates', () => {
    expect(ARC_FACE_TEMPLATE).toHaveLength(5);
    expect(ARC_FACE_TEMPLATE[0]).toEqual([38.2946, 51.6963]);
    const sum = ARC_FACE_TEMPLATE.flat().reduce((acc, value) => acc + value, 0);
    expect(sum).toBeCloseTo(38.2946 + 51.6963 + 73.5318 + 51.5014 + 56.0252 + 71.7366 + 41.5493 + 92.3655 + 70.7299 + 92.2041);
  });
});
