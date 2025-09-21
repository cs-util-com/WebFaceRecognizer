const ORT_VERSION = '1.22.0';
const CDN_JSDELIVR_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;

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
    // Try CDN fallback URL (jsDelivr)
    try {
  const fallbackUrl = `${CDN_JSDELIVR_BASE}/ort.webgpu.min.mjs`;
      if (typeof console !== 'undefined' && console.debug) {
        console.debug('[loadOrt] Retrying WebGPU import from jsDelivr:', fallbackUrl);
      }
      const ort = await dynamicImport(fallbackUrl);
      if (typeof requestAdapter === 'function') {
        try {
          await requestAdapter();
          if (typeof console !== 'undefined' && console.info) {
            console.info('[loadOrt] WebGPU available via jsDelivr; selecting executionProviders=["webgpu"]');
          }
          return { ort, executionProviders: ['webgpu'] };
        } catch (adapterError) {
          if (typeof console !== 'undefined' && console.warn) {
            console.warn('[loadOrt] WebGPU adapter request failed (jsDelivr path):', adapterError);
          }
          errors.push(adapterError);
        }
      }
    } catch (fallbackErr) {
      if (typeof console !== 'undefined' && console.warn) {
        console.warn('[loadOrt] WebGPU import fallback failed:', fallbackErr);
      }
      errors.push(fallbackErr);
    }
  }
  return null;
}

async function loadOrtRuntime(options = {}) {
  const { preferWebGPU = true, navigatorObj, importer } = options;
  const navigatorLike = resolveNavigator(navigatorObj);
  const dynamicImport = importer || ((specifier) => import(specifier));
  const errors = [];

  // Ensure requestAdapter is bound to the GPU object to prevent Illegal invocation
  const requestAdapter = (navigatorLike && navigatorLike.gpu && typeof navigatorLike.gpu.requestAdapter === 'function')
    ? navigatorLike.gpu.requestAdapter.bind(navigatorLike.gpu)
    : undefined;

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
    // Try CDN fallback URL (jsDelivr)
    try {
  const fallbackUrl = `${CDN_JSDELIVR_BASE}/ort.min.mjs`;
      if (typeof console !== 'undefined' && console.debug) {
        console.debug('[loadOrt] Retrying WASM import from jsDelivr:', fallbackUrl);
      }
      const ort = await dynamicImport(fallbackUrl);
      if (typeof console !== 'undefined' && console.info) {
        console.info('[loadOrt] Loaded onnxruntime-web from jsDelivr (WASM); executionProviders=["wasm"]');
      }
      return { ort, executionProviders: ['wasm'] };
    } catch (fallbackErr) {
      errors.push(fallbackErr);
      const message = errors.map((e) => (e && e.message ? e.message : String(e))).join('; ');
      const finalMessage = `Failed to load ONNX Runtime Web: ${message}`;
      if (typeof console !== 'undefined' && console.error) {
        console.error('[loadOrt] ' + finalMessage, { errors });
      }
      throw new Error(finalMessage);
    }
  }
}

export { loadOrtRuntime };
