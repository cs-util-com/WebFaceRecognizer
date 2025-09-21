function resolveNavigator(candidate) {
  if (candidate !== undefined) {
    return candidate;
  }
  return typeof navigator !== 'undefined' ? navigator : undefined;
}

async function tryWebGpu(dynamicImport, requestAdapter, errors) {
  try {
    const ort = await dynamicImport('onnxruntime-web/webgpu');
    if (typeof requestAdapter === 'function') {
      try {
        await requestAdapter();
        return { ort, executionProviders: ['webgpu'] };
      } catch (adapterError) {
        errors.push(adapterError);
      }
    }
  } catch (err) {
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
    const webGpuResult = await tryWebGpu(dynamicImport, requestAdapter, errors);
    if (webGpuResult) {
      return webGpuResult;
    }
  }

  try {
    const ort = await dynamicImport('onnxruntime-web');
    return { ort, executionProviders: ['wasm'] };
  } catch (err) {
    errors.push(err);
    const message = errors.map((e) => (e && e.message ? e.message : String(e))).join('; ');
    throw new Error(`Failed to load ONNX Runtime Web: ${message}`);
  }
}

export { loadOrtRuntime };
