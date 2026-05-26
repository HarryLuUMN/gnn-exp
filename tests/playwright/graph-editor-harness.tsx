import React from "react";
import { createRoot } from "react-dom/client";
import GraphEditor from "../../js/graph_editor/GraphEditor";
import "../../js/graph_editor/style.css";
import type { LinkDatum, NodeDatum } from "../../js/dual_views/dualViewTypes";
import { layoutGraph } from "../../js/dual_views/useGraphScene";

declare global {
  interface Window {
    __GRAPH_EDITOR_EXPECTED?: { id: string; x: number; y: number }[];
    __GRAPH_EDITOR_POSITIONS?: { id: string; x: number; y: number }[];
    __GRAPH_EDITOR_READY?: boolean;
    __GRAPH_EDITOR_LAST_GRAPH?: unknown;
  }
}

const graphData = {
  x: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
  edge_index: [
    [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6],
    [1, 2, 3, 4, 5, 6, 7, 0, 4, 6, 0, 2],
  ],
  edge_attr: Array.from({ length: 12 }, () => [1]),
  y: [0],
  batch: Array(8).fill(0),
};

const layoutNodes: NodeDatum[] = graphData.x.map((_features, id) => ({
  id,
  element: String(id),
}));
const layoutLinks: LinkDatum[] = graphData.edge_index[0].map((source, index) => ({
  source,
  target: graphData.edge_index[1][index],
  attr: graphData.edge_attr[index],
}));

window.__GRAPH_EDITOR_EXPECTED = layoutGraph(
  layoutNodes,
  layoutLinks,
  640,
  640,
  60
).map((node) => ({
  id: `N${node.id}`,
  x: node.x,
  y: node.y,
}));
window.__GRAPH_EDITOR_READY = false;

function Harness() {
  return (
    <GraphEditor
      dataFile={undefined}
      initialGraphData={graphData}
      handleSimulatedGraphChange={(value) => {
        window.__GRAPH_EDITOR_LAST_GRAPH = value;
      }}
      onNodePositionsChange={(positions) => {
        window.__GRAPH_EDITOR_POSITIONS = positions;
        window.__GRAPH_EDITOR_READY = positions.length === graphData.x.length;
      }}
    />
  );
}

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Missing root element");
}

createRoot(rootElement).render(<Harness />);
