const { FaceEmbeddingStore } = require('./faceStore.js');

describe('FaceEmbeddingStore', () => {
  test('enrolls, lists and matches embeddings', () => {
    const store = new FaceEmbeddingStore({ matchThreshold: 0.3 });
    store.enroll('alice', [1, 0, 0]);
    store.enroll('bob', [0, 1, 0], { note: 'second' });
    expect(store.list()).toEqual([
      { id: 'alice', count: 1 },
      { id: 'bob', count: 1 },
    ]);
    const match = store.match([1, 0, 0]);
    expect(match.id).toBe('alice');
    expect(match.score).toBeCloseTo(1);
    expect(store.match([-1, 0, 0])).toBeNull();
    store.remove('bob');
    expect(store.list()).toHaveLength(1);
    store.clear();
    expect(store.list()).toHaveLength(0);
  });

  test('validates thresholds', () => {
    const store = new FaceEmbeddingStore();
    expect(() => store.setThreshold(2)).toThrow();
    expect(() => store.setThreshold(-1)).toThrow();
    expect(() => store.enroll('', [1, 0, 0])).toThrow('id is required');
    const threshold = store.calibrateThreshold([0.8, 0.85], [0.1, 0.2], 0.5);
    expect(threshold).toBeGreaterThan(0.2);
  });
});
