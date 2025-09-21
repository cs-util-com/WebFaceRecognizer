const { loadOrtRuntime } = require('./loadOrt.js');

describe('loadOrtRuntime', () => {
  test('prefers WebGPU when adapter resolves', async () => {
    const webgpuModule = { kind: 'webgpu' };
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web/webgpu') {
        return Promise.resolve(webgpuModule);
      }
      return Promise.reject(new Error('unexpected import'));
    });
    const navigatorObj = { gpu: { requestAdapter: jest.fn().mockResolvedValue({}) } };
    const result = await loadOrtRuntime({ importer, navigatorObj });
    expect(result.executionProviders).toEqual(['webgpu']);
    expect(result.ort).toBe(webgpuModule);
    expect(navigatorObj.gpu.requestAdapter).toHaveBeenCalled();
  });

  test('falls back to WASM when WebGPU fails', async () => {
    const wasmModule = { kind: 'wasm' };
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web/webgpu') {
        return Promise.reject(new Error('no webgpu'));
      }
      if (specifier === 'onnxruntime-web') {
        return Promise.resolve(wasmModule);
      }
      return Promise.reject(new Error('unexpected import'));
    });
    const result = await loadOrtRuntime({ importer, navigatorObj: {} });
    expect(result.executionProviders).toEqual(['wasm']);
    expect(result.ort).toBe(wasmModule);
  });

  test('skips WebGPU when navigator lacks GPU', async () => {
    const wasmModule = { kind: 'wasm' };
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web/webgpu') {
        return Promise.resolve({});
      }
      if (specifier === 'onnxruntime-web') {
        return Promise.resolve(wasmModule);
      }
      return Promise.reject(new Error('unexpected import'));
    });
    const result = await loadOrtRuntime({ importer, navigatorObj: {} });
    expect(result.ort).toBe(wasmModule);
  });

  test('allows explicitly disabling WebGPU preference', async () => {
    const wasmModule = { kind: 'wasm' };
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web') {
        return Promise.resolve(wasmModule);
      }
      return Promise.reject(new Error('unexpected import'));
    });
    const result = await loadOrtRuntime({ importer, preferWebGPU: false });
    expect(importer).toHaveBeenCalledWith('onnxruntime-web');
    expect(result.executionProviders).toEqual(['wasm']);
  });

  test('falls back when adapter rejects', async () => {
    const wasmModule = { kind: 'wasm' };
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web/webgpu') {
        return Promise.resolve({});
      }
      if (specifier === 'onnxruntime-web') {
        return Promise.resolve(wasmModule);
      }
      return Promise.reject(new Error('unexpected import'));
    });
    const navigatorObj = { gpu: { requestAdapter: jest.fn().mockRejectedValue(new Error('no adapter')) } };
    const result = await loadOrtRuntime({ importer, navigatorObj });
    expect(navigatorObj.gpu.requestAdapter).toHaveBeenCalled();
    expect(result.ort).toBe(wasmModule);
  });

  test('uses undefined navigator when global is absent', async () => {
    const originalNavigator = global.navigator;
    global.navigator = undefined;
    try {
      const wasmModule = { kind: 'wasm' };
      const importer = jest.fn((specifier) => {
        if (specifier === 'onnxruntime-web') {
          return Promise.resolve(wasmModule);
        }
        return Promise.reject(new Error('skip webgpu'));
      });
      const result = await loadOrtRuntime({ importer, preferWebGPU: false });
      expect(result.executionProviders).toEqual(['wasm']);
    } finally {
      global.navigator = originalNavigator;
    }
  });

  test('ignores GPU path when requestAdapter is missing', async () => {
    const wasmModule = { kind: 'wasm' };
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web/webgpu') {
        return Promise.resolve({});
      }
      if (specifier === 'onnxruntime-web') {
        return Promise.resolve(wasmModule);
      }
      return Promise.reject(new Error('unexpected import'));
    });
    const navigatorObj = { gpu: {} };
    const result = await loadOrtRuntime({ importer, navigatorObj });
    expect(result.ort).toBe(wasmModule);
  });

  test('throws when both imports fail', async () => {
    const importer = jest.fn(() => Promise.reject(new Error('boom')));
    await expect(loadOrtRuntime({ importer, navigatorObj: {} })).rejects.toThrow('Failed to load ONNX Runtime Web');
  });

  test('aggregates mixed error messages when runtime loading fails', async () => {
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web/webgpu') {
        return Promise.reject('no webgpu');
      }
      return Promise.reject(new Error('no wasm'));
    });
    await expect(loadOrtRuntime({ importer })).rejects.toThrow('no webgpu; no wasm');
  });

  test('accepts explicit null navigator overrides', async () => {
    const wasmModule = { kind: 'wasm' };
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web') {
        return Promise.resolve(wasmModule);
      }
      return Promise.reject(new Error('skip webgpu'));
    });
    const result = await loadOrtRuntime({ importer, navigatorObj: null, preferWebGPU: false });
    expect(result.executionProviders).toEqual(['wasm']);
    expect(result.ort).toBe(wasmModule);
  });

  test('uses CDN WebGPU fallback when direct import fails', async () => {
    const webgpuModule = { kind: 'webgpu-cdn' };
    const fallbackUrl = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web/webgpu') {
        return Promise.reject(new Error('no webgpu direct'));
      }
      if (specifier === fallbackUrl) {
        return Promise.resolve(webgpuModule);
      }
      if (specifier === 'onnxruntime-web') {
        return Promise.reject(new Error('should not reach wasm import'));
      }
      return Promise.reject(new Error('unexpected import'));
    });
    const navigatorObj = { gpu: { requestAdapter: jest.fn().mockResolvedValue({}) } };
    const result = await loadOrtRuntime({ importer, navigatorObj });
    expect(result.executionProviders).toEqual(['webgpu']);
    expect(result.ort).toBe(webgpuModule);
    expect(importer).toHaveBeenCalledWith(fallbackUrl);
    expect(navigatorObj.gpu.requestAdapter).toHaveBeenCalled();
  });

  test('uses CDN WASM fallback when main import fails', async () => {
    const wasmModule = { kind: 'wasm-cdn' };
    const fallbackUrl = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.mjs';
    const importer = jest.fn((specifier) => {
      if (specifier === 'onnxruntime-web') {
        return Promise.reject(new Error('no wasm direct'));
      }
      if (specifier === fallbackUrl) {
        return Promise.resolve(wasmModule);
      }
      return Promise.reject(new Error('unexpected import'));
    });
    const result = await loadOrtRuntime({ importer, preferWebGPU: false });
    expect(result.executionProviders).toEqual(['wasm']);
    expect(result.ort).toBe(wasmModule);
    expect(importer).toHaveBeenCalledWith(fallbackUrl);
  });
});
