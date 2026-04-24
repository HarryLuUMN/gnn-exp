import type { ResolvedRenderer } from "../renderers/capabilities";
import {
  buildVisualizationScene,
  type StaticVisualizationScene,
} from "./renderers/staticScene";
import { renderWebglStaticVisualization } from "./renderers/webglStaticRenderer";
import {
  renderWebgpuStaticVisualization,
} from "./renderers/webgpuStaticRenderer";
import { attachStaticHoverOverlay } from "./renderers/staticHoverOverlay";

export type StaticGpuRenderer = Exclude<ResolvedRenderer, "svg">;

export type StaticPipelineResult =
  | {
      cleanup: () => void;
      scene: StaticVisualizationScene;
    }
  | {
      error: string;
    };

function createCanvas(scene: StaticVisualizationScene) {
  const canvas = document.createElement("canvas");
  canvas.className = "gnn-static-gpu-canvas";
  canvas.width = Math.max(1, Math.ceil(scene.width));
  canvas.height = Math.max(1, Math.ceil(scene.height));
  canvas.style.display = "block";
  canvas.style.width = `${Math.max(1, Math.ceil(scene.width))}px`;
  canvas.style.height = `${Math.max(1, Math.ceil(scene.height))}px`;
  return canvas;
}

export async function staticVisualizationPipeline(
  container: HTMLDivElement,
  renderer: StaticGpuRenderer,
  cellWidth: number,
  cellHeight: number,
  adjacencyMatrix: number[][],
  intmData: any,
  modelInfo: any,
  linkList: any[],
  queries: number[][],
  subgraphSample: boolean,
  mode: string
): Promise<StaticPipelineResult> {
  container.replaceChildren();

  const scene = buildVisualizationScene(
    cellWidth,
    cellHeight,
    adjacencyMatrix,
    intmData,
    linkList,
    queries,
    !!subgraphSample,
    mode
  );
  const canvas = createCanvas(scene);
  container.append(canvas);

  const renderResult =
    renderer === "webgl"
      ? renderWebglStaticVisualization(canvas, scene)
      : await renderWebgpuStaticVisualization(canvas, scene);

  if ("error" in renderResult) {
    canvas.remove();
    return renderResult;
  }

  const hoverOverlay = attachStaticHoverOverlay({
    canvas,
    scene,
    adjacencyMatrix,
    queries,
    mode,
    intmData,
    modelInfo,
    cellWidth,
  });

  return {
    scene,
    cleanup() {
      hoverOverlay.destroy();
      renderResult.destroy();
    },
  };
}
