import type { ResolvedRenderer } from "../renderers/capabilities";
import {
  buildVisualizationScene,
  type StaticExpansionState,
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

  let currentScene: StaticVisualizationScene | null = null;
  let currentRenderDestroy: (() => void) | null = null;
  let currentHoverDestroy: (() => void) | null = null;
  let currentExpansion: StaticExpansionState = null;
  let renderToken = 0;

  const renderScene = async (expansion: StaticExpansionState) => {
    const token = ++renderToken;
    currentHoverDestroy?.();
    currentRenderDestroy?.();
    container.replaceChildren();

    const scene = buildVisualizationScene(
      cellWidth,
      cellHeight,
      adjacencyMatrix,
      intmData,
      linkList,
      queries,
      !!subgraphSample,
      mode,
      { expansion }
    );
    const canvas = createCanvas(scene);
    container.append(canvas);

    const renderResult =
      renderer === "webgl"
        ? renderWebglStaticVisualization(canvas, scene)
        : await renderWebgpuStaticVisualization(canvas, scene);

    if (token !== renderToken) {
      if (!("error" in renderResult)) {
        renderResult.destroy();
      }
      return { stale: true as const };
    }

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
      expandedFeature: expansion,
      onExpansionChange(nextExpansion) {
        currentExpansion = nextExpansion;
        void renderScene(nextExpansion);
      },
    });

    currentScene = scene;
    currentRenderDestroy = renderResult.destroy;
    currentHoverDestroy = hoverOverlay.destroy;
    return { scene };
  };

  const initialRender = await renderScene(currentExpansion);
  if ("error" in initialRender) {
    return initialRender;
  }
  if ("stale" in initialRender) {
    return { error: "Static GPU render was superseded before initialization completed." };
  }

  return {
    scene: currentScene ?? initialRender.scene,
    cleanup() {
      renderToken += 1;
      currentHoverDestroy?.();
      currentRenderDestroy?.();
    },
  };
}
