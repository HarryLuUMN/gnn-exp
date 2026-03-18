import React, { useEffect, useState } from "react";
import GraphView from "./GraphView";
import MatrixView from "./MatrixView";
import type { HoverState, LinkDatum, NodeDatum } from "./dualViewTypes";
import { dualViewVisualizerStyle } from "../utils/const";
import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
// import GraphEditor from "./GraphEditor";

interface Props {
  graphData: any;
  hubNodeA?: number;
  hubNodeB?: number;
  modelType?: string; // "node prediction" | "link prediction" | "graph"
  simulatedGraphData?: any;
  sandboxMode?: boolean;
  nodePositions?: { id: string; x: number; y: number }[];
  onNodePositionChange?: (positions: { id: string; x: number; y: number }[]) => void;
  handleSimulatedGraphChange: any,
  handleNodePositionsChange: any,
}

const DualViews: React.FC<Props> = ({
  graphData,
  hubNodeA,
  hubNodeB,
  modelType,
  simulatedGraphData,
  sandboxMode = false,
  onNodePositionChange,
}) => {
  const [nodes, setNodes] = useState<NodeDatum[]>([]);
  const [links, setLinks] = useState<LinkDatum[]>([]);
  const [hover, setHover] = useState<HoverState>(null);

  const styles = dualViewVisualizerStyle;

  // load data
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = graphData;
        console.log("DualViews received graphData =", graphData, data);

        if (cancelled) return;

        // const isTwitchData = typeof dataFile === "string" && dataFile.includes("twitch.json");

        // determine processed nodes
        const { nodeList, linkList} = preMatrixVisualizationDataProcessingPipe(modelType, hubNodeA, hubNodeB, data, sandboxMode);

        setNodes(nodeList);
        setLinks(linkList);
      } catch (e) {
        console.error("Error loading graph data", e);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [graphData, modelType, sandboxMode, simulatedGraphData, hubNodeA, hubNodeB]);

  const onGraphPositions = (positions: { id: number; x: number; y: number }[]) => {
    if (!onNodePositionChange) return;
    onNodePositionChange(positions.map((p) => ({ id: String(p.id), x: p.x, y: p.y })));
  };

  return (
    <>
      <style>{styles}</style>
      <div style={{ display: "grid", gridTemplateColumns: "550px 550px", gap: 2 }}>
        <div style={{
          width: "67%",
          transform: "scale(0.67)",
          transformOrigin: "top left"
        }}>
          <GraphView
            nodes={nodes}
            links={links}
            linkPredictionMode={!!modelType?.includes("link prediction")}
            onNodePositionChange={onGraphPositions}
            onHover={setHover}
            hover={hover}
          />
        </div>
        <div style={{
          width: "67%",
          transform: "scale(0.67)",
          transformOrigin: "top left"
        }}>
          <MatrixView 
            nodes={nodes}
            links={links}
            onHover={setHover}
            hover={hover}
          />
        </div>
      </div>
    </>
  );
};

export default DualViews;
