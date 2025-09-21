const { canvasToCHWFloat32, normalizeEmbedding } = require('./canvas.js');

function createCanvas(width, height, fill = [100, 110, 120, 255]) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    data[i * 4] = fill[0];
    data[i * 4 + 1] = fill[1];
    data[i * 4 + 2] = fill[2];
    data[i * 4 + 3] = fill[3];
  }
  const context = {
    getImageData: jest.fn(() => ({ data })),
  };
  const canvas = {
    width,
    height,
    getContext: jest.fn(() => context),
  };
  return { canvas, context, data };
}

describe('canvas preprocessing', () => {
  test('converts RGBA canvas into CHW Float32', () => {
    const { canvas } = createCanvas(2, 1, [255, 0, 0, 255]);
    const tensor = canvasToCHWFloat32(canvas, { mean: 0, scale: 1 });
    expect(Array.from(tensor)).toEqual([255, 255, 0, 0, 0, 0]);
  });

  test('normalizes embedding vectors', () => {
    const normalized = normalizeEmbedding([3, 4]);
    expect(normalized[0]).toBeCloseTo(0.6);
    expect(normalized[1]).toBeCloseTo(0.8);
  });

  test('throws when canvas is missing', () => {
    expect(() => canvasToCHWFloat32(null)).toThrow('canvas');
  });
});
