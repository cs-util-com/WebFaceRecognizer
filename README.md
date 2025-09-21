# **In-Browser Face Recognition Prototype (ArcFace \+ SCRFD) — Developer Guide**

[**https://chatgpt.com/g/g-p-685da6143e6c8191beaaf9db08639d28-better-answers/c/68c0319f-4334-8330-9b72-4efb9c9cda6f**](https://chatgpt.com/g/g-p-685da6143e6c8191beaaf9db08639d28-better-answers/c/68c0319f-4334-8330-9b72-4efb9c9cda6f) 

**Goal:** Ship a private, entirely client-side FaceID prototype that runs in the browser (no server), using **ArcFace** for embeddings and **SCRFD** for detection+5-point landmarks.  
 **Constraints:** Commercial-friendly licensing, modern-browser performance, and clean upgrade path.

---

## **1\) Architecture (high level)**

1. **Frame source** → HTML `<video>` / `<img>` / `<canvas>`.

2. **Face detection \+ 5 landmarks** → **SCRFD** (ONNX) via **ONNX Runtime Web** (WebGPU if available, otherwise WASM). Output: face bboxes \+ 5 keypoints. ([insightface.ai](https://insightface.ai/scrfd?utm_source=chatgpt.com), [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/object-detection-pre-trained-model-inference-issue-in-deepstream/299427?utm_source=chatgpt.com))

3. **Alignment** → Similarity transform to 112×112 using ArcFace’s 5-point reference template. ([GitHub](https://github.com/deepinsight/insightface/issues/1154?utm_source=chatgpt.com))

4. **Embedding** → **ArcFace ResNet100 (ONNX)**, 512-D L2-normalized vector. Compare with cosine similarity. ([docs.openvino.ai](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html))

5. **Match** → In-memory store of (id, embedding). For a prototype, brute-force cosine; later, swap in ANN.

---

## **2\) Models & licenses**

* **ArcFace ResNet100 (ONNX)**: OpenVINO Model Zoo distribution (Apache-2.0).  
   *Input:* `1×3×112×112` (RGB for the “original model”), *Output:* `1×512` embedding. “Comparable in cosine distance.” ([docs.openvino.ai](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html))

* **SCRFD (ONNX)**: Detector with 5 keypoints per face; widely used and fast. Choose an ONNX export that clearly includes keypoints (e.g., `scrfd_2.5g*_kps`/`bnkps`). **Be careful with weights licensing**: InsightFace site notes many pretrained weights are for **non-commercial research**. If you need commercial usage, use an Apache-2.0–friendly source/export or train your own. ([PyPI](https://pypi.org/project/insightface/?utm_source=chatgpt.com), [GitHub](https://github.com/cospectrum/scrfd?utm_source=chatgpt.com))

**Pragmatic fallback (for faster bring-up):** OpenCV YuNet (ONNX) also outputs **5 landmarks** and has straightforward post-processing. You can swap it in for step (2) if SCRFD post-processing feels heavy for day-1. ([docs.opencv.org](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html?utm_source=chatgpt.com))

---

## **3\) Browser runtime (ONNX Runtime Web)**

Prefer **WebGPU EP** for speed; fall back to **WASM**. Import path & EP selection:

 // WebGPU first, fallback to WASM  
let ort;  
try {  
  ort \= await import('onnxruntime-web/webgpu');  
  await navigator.gpu.requestAdapter(); // throws if unsupported  
  var ep \= \['webgpu'\];  
} catch {  
  ort \= await import('onnxruntime-web'); // WASM  
  var ep \= \['wasm'\];  
}  
const session \= await ort.InferenceSession.create(modelUrl, { executionProviders: ep });

*  ONNX Runtime Web’s WebGPU how-to and EP flags are documented here. ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html))

**WASM threading & SIMD:** For multi-threaded WASM, your app must be **cross-origin isolated** (COOP/COEP headers) or ORT will silently degrade to single-threaded. You can also set `ort.env.wasm.numThreads`. ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html?utm_source=chatgpt.com))

 Cross-Origin-Opener-Policy: same-origin  
Cross-Origin-Embedder-Policy: require-corp

*  (Expect console warnings if `numThreads>1` without isolation.) ([GitHub](https://github.com/microsoft/onnxruntime/issues/19148?utm_source=chatgpt.com), [app.semanticdiff.com](https://app.semanticdiff.com/gh/microsoft/onnxruntime/pull/19179/overview?utm_source=chatgpt.com))

* **Deploying the right WASM binary:** When using WebGPU/WebNN, ORT loads `ort-wasm-simd-threaded.jsep.wasm`; otherwise `ort-wasm-simd-threaded.wasm`. If hosting assets under a custom path, set `ort.env.wasm.wasmPaths = '/static/onnx/'`. ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/deploy.html?utm_source=chatgpt.com), [Stack Overflow](https://stackoverflow.com/questions/77179151/onnxruntime-web-fails-to-find-ort-wasm-simd-wasm-doesnt-use-my-static-folder?utm_source=chatgpt.com))

---

## **4\) Project skeleton**

/public  
  /models  
    arcface\_r100.onnx  
    scrfd\_2.5g\_kps\_640x640.onnx  
  /ort  
    ort-webgpu.esm.min.js (optional)  
    ort-wasm-simd-threaded\*.wasm / \*.jsep.wasm  
/src  
  app.js  
  align.js  
  preprocess.js  
index.html  
server.js (express or any static server with COOP/COEP)

---

## **5\) Detection (SCRFD) — inputs & post-processing**

**Inputs:** Many SCRFD ONNX exports take `1×3×640×640` (RGB). You’ll typically **letterbox** the source to 640×640, keep track of scale/offset, and later un-scale boxes/landmarks to original coordinates.

**Outputs:** 9 tensors (per stride `8,16,32`): scores, bbox deltas, and 5-point keypoints (10 numbers) for each anchor. Shapes look like `(12800,1)`, `(12800,4)`, `(12800,10)` at 640×640 for stride-8, plus analogous for 16 & 32\. You must decode deltas, apply (sigmoid) confidences, then **NMS**. ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/object-detection-pre-trained-model-inference-issue-in-deepstream/299427?utm_source=chatgpt.com))

Tip: Implementation recipes and output ordering discussions are scattered; confirm output order for your exact model (there are variants). ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/tx2-scrfd-model-tensorrt-conversion-faulty/325077?utm_source=chatgpt.com))

**Minimal decode loop (pseudo-JS):**

// after running session.run(feeds)  
const decoded \= \[\];  
for (const stride of \[8,16,32\]) {  
  const scores \= out\[\`scores\_${stride}\`\];  // Float32Array  
  const bboxes \= out\[\`bboxes\_${stride}\`\];  // Float32Array  
  const kps    \= out\[\`kps\_${stride}\`\];     // Float32Array  
  // 1\) iterate over locations; 2\) filter score \> confThresh;  
  // 3\) decode bbox deltas to pixel coords; 4\) decode 5 keypoints (x,y)\*5;  
  // 5\) push candidate {bbox, kps, score}  
}  
const faces \= nms(decoded, 0.4);

If you want a quicker path, use **YuNet** for step-(2): its output is a simple `N×15` array (bbox \+ 5 landmarks). ([docs.opencv.org](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html?utm_source=chatgpt.com))

---

## **6\) Alignment (the critical “gotcha”)**

ArcFace expects **aligned** crops. Use the **5-point similarity transform** mapping the detected landmarks to the **ArcFace reference template** (for a 112×112 crop):

\[\[38.2946, 51.6963\],  
 \[73.5318, 51.5014\],  
 \[56.0252, 71.7366\],  
 \[41.5493, 92.3655\],  
 \[70.7299, 92.2041\]\]

These are the canonical InsightFace/ArcFace 5-point reference coordinates. ([GitHub](https://github.com/deepinsight/insightface/issues/1154?utm_source=chatgpt.com))

**Implementation notes:**

* Sort your detected keypoints in order: **right eye, left eye, nose tip, right mouth corner, left mouth corner** (verify the order your detector emits). Mismatched ordering silently destroys accuracy. ([GitHub](https://github.com/deepinsight/insightface/issues/2759?utm_source=chatgpt.com))

* Compute a **similarity transform (scale/rotation/translation)** via least-squares (Procrustes) and warp into a 112×112 canvas.

* You can implement the warp using an offscreen `<canvas>`: set a 2×3 transform matrix and draw the source image.

**Sample (fit-to-template)**

import {estimateSimilarity} from './align.js'; // returns 2x3 matrix  
const M \= estimateSimilarity(srcKps, ARC\_TEMPLATE\_5PTS);  
const alignedCanvas \= warpTo112(inputImageBitmap, M);

---

## **7\) Embeddings (ArcFace R100 ONNX)**

**Input contract:** `1×3×112×112`, **RGB** (for the “original” ONNX). Output is a `1×512` embedding; compare with **cosine** (after L2-norm). ([docs.openvino.ai](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html))

**Preprocessing (typical):**

1. Read pixels from the aligned 112×112 canvas (RGBA → **RGB**).

2. Convert to `Float32Array`, **CHW** layout.

3. Normalize to approximately **\[-1, 1\]** via `(x - 127.5) / 128.0`. (This is the common InsightFace/ArcFace preprocessing; confirm for your chosen weights.) ([GitHub](https://github.com/deepinsight/insightface/issues/2660?utm_source=chatgpt.com))

**Run \+ normalize:**

const input \= new ort.Tensor('float32', chwFloat32, \[1,3,112,112\]);  
const {fc1} \= await arcfaceSession.run({data: input}); // 'fc1' is typical output name  
// L2 normalize  
const v \= fc1.data;  
let n=0; for (let i=0;i\<512;i++) n \+= v\[i\]\*v\[i\];  
const inv \= 1/Math.sqrt(n);  
for (let i=0;i\<512;i++) v\[i\]\*=inv;

(The ArcFace ONNX in OMZ documents the shapes; “cosine distance” comparability is expected.) ([docs.openvino.ai](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html))

---

## **8\) Matching logic**

**Brute force, prototype:** Compute cosine similarity to every enrolled vector.

 function cosine(a,b){let s=0; for(let i=0;i\<a.length;i++) s+=a\[i\]\*b\[i\]; return s;}

*   
* **Threshold:** Do **not** hard-code a universal value. Calibrate by collecting a small validation set of *positives* (same person) and *negatives* and plot similarity distributions; pick the operating point for your desired FAR (e.g., 1e-3). LFW-style setups evaluate with cosine on L2-normalized embeddings, but deployment thresholds depend on camera, lighting, and your alignment quality. (ArcFace LFW report is near-saturated accuracy, but *your* pipeline will vary.) ([docs.openvino.ai](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html))

* **Future:** Switch to ANN for scale; e.g., a Wasm/JS HNSW library when identities grow.

---

## **9\) Performance checklist**

* **Prefer WebGPU** EP when available; toggle ORT’s WebGPU features if needed (graph capture, IO binding). ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html))

* **For WASM:** enable SIMD & threads (**requires** cross-origin isolation). Tune `ort.env.wasm.numThreads`. ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html?utm_source=chatgpt.com))

* **Workers:** Run detection & embedding in a **Web Worker**; keep the UI thread free.

* **Resolution gating:** Downscale the input frame (e.g., 720p→640px letterbox for detector).

* **Cache models:** Preload and cache ONNX \+ WASM binaries; serve with long-lived cache headers.

* **Session reuse:** Create inference sessions once; reuse feeds/buffers to minimize GC.

* **NMS:** Vectorize and early-reject by score to cut candidates.

---

## **10\) Security, privacy, and ethics**

* **All local.** This design does inference **entirely in-browser** (no image upload). ORT Web explicitly supports this flow. ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/))

* Store embeddings in memory (and optionally in IndexedDB) encrypted if you persist them.

* **Spoofing risk:** Printed/photo attacks can confuse FR (see adversarial examples, e.g., **AdvHat**). Add liveness later (blink/3D/IR). ([arXiv](https://arxiv.org/abs/1908.08705?utm_source=chatgpt.com))

* **Bias:** Thresholds and performance can vary across demographics; validate on your population.

---

## **11)End-to-end “first prototype” steps**

1. **Serve with COOP/COEP** (enables WASM threads; also useful for WebGPU). [onnxruntime.ai](http://onnxruntime.ai)

2. **Install deps**  
    `npm i onnxruntime-web`

3. **Host models** under `/public/models` (same origin to avoid CORS headaches). If you must host elsewhere, ensure CORS and `Cross-Origin-Resource-Policy: cross-origin`.

4. **Load runtime & sessions**  
   `import * as ort from 'onnxruntime-web/webgpu'; // fallback to 'onnxruntime-web' if needed`  
   `// Optionally set wasm paths if self-hosted under /ort`  
   `// ort.env.wasm.wasmPaths = '/ort/';`  
  `const scrfd = await ort.InferenceSession.create('/models/scrfd_2.5g_bnkps.onnx', {executionProviders:['webgpu']});`  
  `const arc  = await ort.InferenceSession.create('/models/arcfaceresnet100-11-int8.onnx', {executionProviders:['webgpu']});`

5. **Detect** (SCRFD)  
   * Letterbox to 640×640; build `Tensor('float32',[1,3,640,640])`; run; decode per-stride outputs; NMS. Typical output bundle contains bboxes and 5 keypoints. [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/object-detection-pre-trained-model-inference-issue-in-deepstream/299427?utm_source=chatgpt.com)

6. **Align**  
   * Use ArcFace 5-point template (above); compute similarity transform; warp to 112×112. [GitHub](https://github.com/deepinsight/insightface/issues/1154?utm_source=chatgpt.com)

7. **Embed & compare**  
   * Preprocess (RGB CHW 112×112; `(x-127.5)/128`), run ArcFace, L2-normalize, cosine compare. (ArcFace ONNX shapes per OMZ docs.) [docs.openvino.ai](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html)

### Using INT8 ArcFace in this repo

This project defaults to the INT8 embedder model (`/models/arcfaceresnet100-11-int8.onnx`). Many INT8 exports accept `uint8` tensors; others still expect `float32` with normalization. You can control this via:

- Query params: `?embedder=/models/arcfaceresnet100-11-int8.onnx&embedderInputType=uint8&embedderInputName=data&embedderOutputName=fc1`
- Or in code: `new FaceRecognitionApp({ embedderModelUrl, embedderInputType: 'uint8', embedderInputName: 'data', embedderOutputName: 'fc1' })`

If your INT8 model expects float inputs, keep `embedderInputType=float32` and optionally change normalization via `normalizeMean` and `normalizeScale`.

Caveats:
- Some quantized ops may not be accelerated by WebGPU; runtime may fall back to WASM EP.
- Slight accuracy drop vs FP32 is common.

8. **Calibrate a threshold**  
   * Capture a handful of positive/negative pairs from your camera and sweep thresholds to choose an operating point.

---

## **12\) Common “gotchas”**

* **Wrong color order:** The **OMZ ArcFace ONNX “original” expects RGB**; don’t feed BGR (many OpenCV snippets are BGR). ([docs.openvino.ai](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html))

* **No alignment \= bad accuracy:** Use the **5-point template**. Even tiny landmark order mistakes will crater performance. ([GitHub](https://github.com/deepinsight/insightface/issues/1154?utm_source=chatgpt.com))

* **SCRFD output order varies:** ONNX exports differ; confirm tensor order and apply the right decode. (Shapes usually match the “scores/bboxes/kps per stride” pattern.) ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/object-detection-pre-trained-model-inference-issue-in-deepstream/299427?utm_source=chatgpt.com))

* **WASM single-threaded unexpectedly:** You didn’t set COOP/COEP or the page isn’t `crossOriginIsolated`. Fix headers. ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html?utm_source=chatgpt.com))

* **WebGPU missing in certain contexts:** Some environments (e.g., service workers/extensions) don’t expose WebGPU; fall back to WASM. ([GitHub](https://github.com/microsoft/onnxruntime/issues/20876?utm_source=chatgpt.com))

* **Model asset path issues:** If ORT can’t find its `.wasm` binaries, set `ort.env.wasm.wasmPaths`. ([Stack Overflow](https://stackoverflow.com/questions/77179151/onnxruntime-web-fails-to-find-ort-wasm-simd-wasm-doesnt-use-my-static-folder?utm_source=chatgpt.com))

* **Thresholds copy-pasted from blogs:** Don’t. Calibrate on your capture stack; lighting & alignment shift distributions.

---

## **13\) Minimal code snippets**

### **13.1 Frame → CHW float32 (normalized)**

function canvasToCHWFloat32(canvas, {mean=127.5, scale=1/128}={}) {  
  const ctx \= canvas.getContext('2d', {willReadFrequently:true});  
  const {width, height} \= canvas;  
  const rgba \= ctx.getImageData(0,0,width,height).data;  
  const size \= width\*height;  
  const arr \= new Float32Array(3\*size);  
  // RGB to CHW  
  for (let i=0, p=0; i\<size; i++, p+=4) {  
    const r \= (rgba\[p\]   \- mean) \* scale;  
    const g \= (rgba\[p+1\] \- mean) \* scale;  
    const b \= (rgba\[p+2\] \- mean) \* scale;  
    arr\[i\]           \= r;          // R (channel 0\)  
    arr\[i \+ size\]    \= g;          // G (channel 1\)  
    arr\[i \+ 2\*size\]  \= b;          // B (channel 2\)  
  }  
  return arr;  
}

### **13.2 Cosine similarity**

function cosineSim(a, b) {  
  let s=0; for (let i=0;i\<a.length;i++) s \+= a\[i\]\*b\[i\];  
  return s; // a & b are L2-normalized  
}

---

## **14\) References (key specs you’ll likely check)**

* **ArcFace R100 ONNX (OMZ):** shapes, RGB vs BGR notes; Apache-2.0. ([docs.openvino.ai](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html))

* **SCRFD overview & kps:** model details; 5 keypoints; sample shapes. ([insightface.ai](https://insightface.ai/scrfd?utm_source=chatgpt.com), [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/object-detection-pre-trained-model-inference-issue-in-deepstream/299427?utm_source=chatgpt.com))

* **ArcFace 5-point reference template:** the canonical 112×112 coordinates. ([GitHub](https://github.com/deepinsight/insightface/issues/1154?utm_source=chatgpt.com))

* **ONNX Runtime Web (WebGPU & WASM threading/flags/deploy):** usage, env flags, jsep wasm. ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html))

* **YuNet (alternative detector) 5 landmarks API:** output includes 5 facial landmarks. ([docs.opencv.org](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html?utm_source=chatgpt.com))

---

## **15\) Roadmap (post-prototype)**

* Liveness (blink/CNN), tracking (Kalman/KLT), per-camera thresholding, on-device enrollment flow, IndexedDB encryption, ANN search, batch inference, ORT ORT-format conversion for faster init.

---

That’s a standalone developer brief to implement a first in-browser prototype with ArcFace+SCRFD, with all the foot-guns called out and links to the specs you’ll actually need.

A few risks to keep in mind:

* **Alignment sensitivity:** Landmark mis-ordering or poor alignment quietly tanks accuracy—budget time for robust tests and visualization. ([GitHub](https://github.com/deepinsight/insightface/issues/1154?utm_source=chatgpt.com))

* **WebGPU variability:** Some enterprise browsers disable it; ensure WASM performance is acceptable or you’ll see jank on mid-tier laptops. ([onnxruntime.ai](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html))