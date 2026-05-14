import React from "react";
import { createRoot } from "react-dom/client";
import GNNVisualizer from "../../js/gnn_visualizer/GNNVisualizer";
import "../../js/gnn_visualizer/style.css";
import type {
  RendererMode,
  ResolvedRenderer,
} from "../../js/renderers/capabilities";

type LayerKind = "gat" | "graphsage" | "gin";

type HarnessData = {
  intmData: Record<string, number[][]>;
  modelInfo: Record<string, unknown>;
};

declare global {
  interface Window {
    __GNN_LAYER_READY?: boolean;
  }
}

const graphData = {
  x: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
  ],
  edge_index: [
    [0, 1, 2, 3, 1],
    [1, 2, 3, 0, 3],
  ],
  y: [0, 1, 0, 1],
};

const baseAct1 = [
  [0.4, -0.2],
  [0.1, 0.5],
  [-0.3, 0.2],
  [0.6, 0.1],
];

function baseClassifier(inputLength: number) {
  return {
    type: "Linear",
    weight: [
      Array.from({ length: inputLength }, (_, index) => 0.2 - index * 0.03),
      Array.from({ length: inputLength }, (_, index) => -0.1 + index * 0.04),
    ],
    bias: [0.05, -0.02],
  };
}

function layerHarnessData(kind: LayerKind): HarnessData {
  if (kind === "gat") {
    const act1 = baseAct1.map(([a, b]) => [a, b, a * 0.5, b * -0.5]);
    return {
      intmData: {
        act0: graphData.x,
        act1,
        softmax: [
          [0.55, 0.45],
          [0.35, 0.65],
          [0.48, 0.52],
          [0.7, 0.3],
        ],
      },
      modelInfo: {
        conv1: {
          type: "GATConv",
          aggregation: "sum",
          heads: 2,
          concat: true,
          weight: [
            [0.2, -0.1, 0.05],
            [0.3, 0.15, -0.2],
            [-0.25, 0.4, 0.1],
            [0.12, -0.18, 0.22],
          ],
          bias: [0.01, -0.02, 0.03, -0.04],
        },
        classifier: baseClassifier(4),
      },
    };
  }

  if (kind === "gin") {
    return {
      intmData: {
        act0: graphData.x,
        act1: baseAct1,
        softmax: [
          [0.58, 0.42],
          [0.4, 0.6],
          [0.52, 0.48],
          [0.62, 0.38],
        ],
      },
      modelInfo: {
        conv1: {
          type: "GINConv",
          aggregation: "sum",
          eps: 0.1,
          weight: [
            [0.35, -0.15, 0.2],
            [-0.1, 0.28, 0.18],
          ],
          bias: [0.03, -0.01],
        },
        classifier: baseClassifier(2),
      },
    };
  }

  return {
    intmData: {
      act0: graphData.x,
      act1: baseAct1,
      softmax: [
        [0.6, 0.4],
        [0.3, 0.7],
        [0.45, 0.55],
        [0.68, 0.32],
      ],
    },
    modelInfo: {
      conv1: {
        type: "SAGEConv",
        aggregation: "mean",
        weight: [
          [0.25, -0.2, 0.15],
          [-0.12, 0.31, 0.08],
        ],
        bias: [0.02, 0.04],
        root_weight: [
          [0.1, 0.05, -0.05],
          [0.04, -0.08, 0.11],
        ],
      },
      classifier: baseClassifier(2),
    },
  };
}

function selectedLayerKind(): LayerKind {
  const value = new URLSearchParams(window.location.search).get("model");
  return value === "gat" || value === "gin" || value === "graphsage"
    ? value
    : "graphsage";
}

const data = layerHarnessData(selectedLayerKind());
const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Missing root element");
}

function Harness() {
  const [effectiveRenderer, setEffectiveRenderer] =
    React.useState<ResolvedRenderer>("svg");
  const renderer: RendererMode = "svg";

  return (
    <GNNVisualizer
      graphData={graphData}
      modelInfo={data.modelInfo}
      onLoadComplete={() => {
        window.__GNN_LAYER_READY = true;
      }}
      intmData={data.intmData}
      renderToken={1}
      queries={[[1, 3]]}
      subgraphSample={false}
      mode="node"
      renderer={renderer}
      effectiveRenderer={effectiveRenderer}
      setEffectiveRenderer={setEffectiveRenderer}
    />
  );
}

createRoot(rootElement).render(<Harness />);
