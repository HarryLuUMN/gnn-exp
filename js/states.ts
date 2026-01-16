let canvasId: string | null = null;

export function initCanvasId() {
  const id = Math.floor(Math.random() * 1000000000);
  canvasId = `canvas-${id}`; 
  return canvasId;
}

export function getCanvasId(): string {
  if (canvasId === null) {
    throw new Error("Canvas ID not initialized");
  }
  return canvasId;
}

let svgId: string | null = null;

export function initSvgId() {
  const id = Math.floor(Math.random() * 1000000000);
  svgId = `svg-${id}`;  
  return svgId;
}

export function getSvgId(): string {
  if (svgId === null) {
    throw new Error("SVG ID not initialized");
  }
  return svgId;
}