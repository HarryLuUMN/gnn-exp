import React from "react";
import { createRoot } from "react-dom/client";
import GNNVisualizer from "../../js/gnn_visualizer/GNNVisualizer";
import "../../js/gnn_visualizer/style.css";
import type {
  RendererMode,
  ResolvedRenderer,
} from "../../js/renderers/capabilities";

type LayerKind = "gcn_logits" | "gat" | "graph_gat" | "graphsage" | "gin" | "large_science_graph";
type HarnessRenderer = "auto" | "svg" | "webgl" | "webgpu";

type HarnessData = {
  graphData: Record<string, unknown>;
  intmData: Record<string, number[][]>;
  modelInfo: Record<string, unknown>;
  queries: number[][];
  subgraphSample: boolean;
  mode: string;
};

declare global {
  interface Window {
    __GNN_LAYER_READY?: boolean;
  }
}

function selectedLayerKind(): LayerKind {
  const value = new URLSearchParams(window.location.search).get("model");
  return value === "gcn_logits" ||
    value === "gat" ||
    value === "graph_gat" ||
    value === "gin" ||
    value === "graphsage" ||
    value === "large_science_graph"
    ? value
    : "graphsage";
}

function selectedRenderer(): HarnessRenderer {
  const value = new URLSearchParams(window.location.search).get("renderer");
  return value === "auto" || value === "svg" || value === "webgl" || value === "webgpu"
    ? value
    : "svg";
}

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Missing root element");
}

function Harness() {
  const layerKind = selectedLayerKind();
  const [data, setData] = React.useState<HarnessData | null>(null);
  const [effectiveRenderer, setEffectiveRenderer] =
    React.useState<ResolvedRenderer>("svg");
  const [viewportHeight, setViewportHeight] = React.useState(820);
  const renderer: RendererMode = selectedRenderer();

  React.useEffect(() => {
    window.__GNN_LAYER_READY = false;
    fetch(`../.cache/fixtures/${layerKind}.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load fixture for ${layerKind}`);
        }
        return response.json() as Promise<HarnessData>;
      })
      .then(setData)
      .catch((error: unknown) => {
        console.error(error);
        throw error;
      });
  }, [layerKind]);

  if (!data) {
    return <div>Loading fixture...</div>;
  }

  return (
    <GNNVisualizer
      graphData={data.graphData}
      modelInfo={data.modelInfo}
      onLoadComplete={() => {
        window.__GNN_LAYER_READY = true;
      }}
      intmData={data.intmData}
      renderToken={1}
      queries={data.queries}
      subgraphSample={data.subgraphSample}
      mode={data.mode}
      renderer={renderer}
      effectiveRenderer={effectiveRenderer}
      setEffectiveRenderer={setEffectiveRenderer}
      viewportHeight={viewportHeight}
      setViewportHeight={setViewportHeight}
      autoFit
    />
  );
}

createRoot(rootElement).render(<Harness />);
