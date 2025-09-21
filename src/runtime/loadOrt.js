function resolveNavigator(candidate) {
  if (candidate !== undefined) {
    return candidate;
  }
  return typeof navigator !== 'undefined' ? navigator : undefined;
}

async function tryWebGpu(dynamicImport, requestAdapter, errors) {
  try {
    // Debug: trying to import WebGPU backend
    if (typeof console !== 'undefined' && console.debug) {
      console.debug('[loadOrt] Attempting to import onnxruntime-web/webgpu');
    }
    const ort = await dynamicImport('onnxruntime-web/webgpu');
    if (typeof requestAdapter === 'function') {
      try {
        if (typeof console !== 'undefined' && console.debug) {
          console.debug('[loadOrt] Requesting GPU adapter via navigator.gpu.requestAdapter()');
        }
        await requestAdapter();
        if (typeof console !== 'undefined' && console.info) {
          console.info('[loadOrt] WebGPU available; selecting executionProviders=["webgpu"]');
        }
        return { ort, executionProviders: ['webgpu'] };
      } catch (adapterError) {
        if (typeof console !== 'undefined' && console.warn) {
          console.warn('[loadOrt] WebGPU adapter request failed:', adapterError);
        }
        errors.push(adapterError);
      }
    }
  } catch (err) {
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('[loadOrt] Failed to import onnxruntime-web/webgpu:', err);
    }
    errors.push(err);
  }
  return null;
}

async function loadOrtRuntime(options = {}) {
  const { preferWebGPU = true, navigatorObj, importer } = options;
  const navigatorLike = resolveNavigator(navigatorObj);
  const dynamicImport = importer || ((specifier) => import(specifier));
  const errors = [];

  const requestAdapter = navigatorLike && navigatorLike.gpu && navigatorLike.gpu.requestAdapter;

  if (preferWebGPU) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug('[loadOrt] preferWebGPU=true; navigator.gpu', !!(navigatorLike && navigatorLike.gpu));
    }
    const webGpuResult = await tryWebGpu(dynamicImport, requestAdapter, errors);
    if (webGpuResult) {
      return webGpuResult;
    }
  }

  try {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug('[loadOrt] Falling back to WASM backend import (onnxruntime-web)');
    }
    const ort = await dynamicImport('onnxruntime-web');
    if (typeof console !== 'undefined' && console.info) {
      console.info('[loadOrt] Loaded onnxruntime-web (WASM); executionProviders=["wasm"]');
    }
    return { ort, executionProviders: ['wasm'] };
  } catch (err) {
    errors.push(err);
    const message = errors.map((e) => (e && e.message ? e.message : String(e))).join('; ');
    const finalMessage = `Failed to load ONNX Runtime Web: ${message}`;
    if (typeof console !== 'undefined' && console.error) {
      console.error('[loadOrt] ' + finalMessage, { errors });
    }
    throw new Error(finalMessage);
  }
}

export { loadOrtRuntime };
