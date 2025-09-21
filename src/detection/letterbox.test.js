const { calculateLetterbox, mapBoxToOriginal, mapKeypointsToOriginal } = require('./letterbox.js');

describe('letterbox utilities', () => {
  test('computes padding and scale', () => {
    const meta = calculateLetterbox(1280, 720, 640);
    expect(meta.scale).toBeCloseTo(0.5);
    expect(meta.padY).toBeCloseTo((640 - 360) / 2);
    expect(meta.padX).toBe(0);
  });

  test('maps boxes and keypoints back to original space', () => {
    const meta = { scale: 0.5, padX: 10, padY: 20 };
    const box = [20, 20, 60, 60];
    const mappedBox = mapBoxToOriginal(box, meta);
    expect(mappedBox).toEqual([20, 0, 100, 80]);
    const points = mapKeypointsToOriginal([[20, 20], [60, 60]], meta);
    expect(points).toEqual([[20, 0], [100, 80]]);
  });

  test('throws when provided invalid dimensions', () => {
    expect(() => calculateLetterbox(0, 100, 640)).toThrow('Source dimensions');
  });
});
