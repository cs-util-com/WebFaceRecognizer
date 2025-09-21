function calculateLetterbox(width, height, targetSize) {
  if (width <= 0 || height <= 0) {
    throw new Error('Source dimensions must be positive');
  }
  const scale = Math.min(targetSize / width, targetSize / height);
  const scaledWidth = width * scale;
  const scaledHeight = height * scale;
  const padX = (targetSize - scaledWidth) / 2;
  const padY = (targetSize - scaledHeight) / 2;
  return {
    scale,
    padX,
    padY,
    scaledWidth,
    scaledHeight,
    targetSize,
  };
}

function mapBoxToOriginal(box, letterboxMeta) {
  const { scale, padX, padY } = letterboxMeta;
  return [
    (box[0] - padX) / scale,
    (box[1] - padY) / scale,
    (box[2] - padX) / scale,
    (box[3] - padY) / scale,
  ];
}

function mapKeypointsToOriginal(keypoints, letterboxMeta) {
  return keypoints.map(([x, y]) => mapBoxToOriginal([x, y, x, y], letterboxMeta).slice(0, 2));
}

export { calculateLetterbox, mapBoxToOriginal, mapKeypointsToOriginal };
