import { cosineSimilarity, l2Normalize } from '../math/vector.js';
import { estimateThreshold } from './thresholdCalibrator.js';

class FaceEmbeddingStore {
  constructor({ matchThreshold = 0.45 } = {}) {
    this.matchThreshold = matchThreshold;
    this._entries = new Map();
  }

  setThreshold(threshold) {
    if (threshold <= -1 || threshold >= 1) {
      throw new Error('Cosine threshold must be within (-1, 1)');
    }
    this.matchThreshold = threshold;
  }

  enroll(id, embedding, metadata = {}) {
    if (!id) {
      throw new Error('An id is required to enroll an embedding');
    }
    const normalized = l2Normalize(Array.from(embedding));
    const record = { id, embedding: normalized, metadata };
    const list = this._entries.get(id) || [];
    list.push(record);
    this._entries.set(id, list);
    return record;
  }

  remove(id) {
    return this._entries.delete(id);
  }

  clear() {
    this._entries.clear();
  }

  list() {
    return Array.from(this._entries.entries()).map(([id, embeddings]) => ({ id, count: embeddings.length }));
  }

  _iterateEmbeddings() {
    return Array.from(this._entries.values()).flat();
  }

  match(queryEmbedding) {
    const query = l2Normalize(Array.from(queryEmbedding));
    let bestMatch = null;
    this._iterateEmbeddings().forEach((candidate) => {
      const score = cosineSimilarity(query, candidate.embedding);
      if (!bestMatch || score > bestMatch.score) {
        bestMatch = { id: candidate.id, score, metadata: candidate.metadata };
      }
    });
    if (bestMatch && bestMatch.score >= this.matchThreshold) {
      return bestMatch;
    }
    return null;
  }

  calibrateThreshold(positives, negatives, targetFAR = 1e-3) {
    const { threshold } = estimateThreshold(positives, negatives, targetFAR);
    this.matchThreshold = threshold;
    return threshold;
  }
}

export { FaceEmbeddingStore };
