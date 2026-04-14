import React, { useEffect, useState } from "react";
import GraphView from "./GraphView";
import MatrixView from "./MatrixView";
import type { HoverState, LinkDatum, NodeDatum } from "./dualViewTypes";
import { dualViewVisualizerStyle } from "../utils/const";
import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import {
  detectCapabilities,
  resolveRenderer,
  type RendererMode,
  type ResolvedRenderer,
} from "./renderers/capabilities";
import { useGraphScene } from "./useGraphScene";

interface Props {
  graphData: any;
  renderer: RendererMode;
  effectiveRenderer: ResolvedRenderer;
  setRenderer: (renderer: RendererMode) => void;
  setEffectiveRenderer: (renderer: ResolvedRenderer) => void;
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
  renderer,
  effectiveRenderer,
  setRenderer,
  setEffectiveRenderer,
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
  const [fallbackFailures, setFallbackFailures] = useState<
    Partial<Record<ResolvedRenderer, string>>
  >({});

  const styles = dualViewVisualizerStyle;
  const graphWidth = 800;
  const graphHeight = 600;
  const padding = 60;
  const capabilities = React.useMemo(() => detectCapabilities(), []);
  const resolution = React.useMemo(
    () => resolveRenderer(renderer, capabilities, fallbackFailures),
    [capabilities, fallbackFailures, renderer]
  );

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

  useEffect(() => {
    if (effectiveRenderer !== resolution.effectiveRenderer) {
      setEffectiveRenderer(resolution.effectiveRenderer);
    }
  }, [
    effectiveRenderer,
    resolution.effectiveRenderer,
    setEffectiveRenderer,
  ]);

  const onBackendFailure = React.useCallback(
    (backend: Exclude<ResolvedRenderer, "svg">, reason: string) => {
      setFallbackFailures((current) => {
        if (current[backend] === reason) {
          return current;
        }

        return { ...current, [backend]: reason };
      });
    },
    []
  );

  const onGraphPositions = (positions: { id: number; x: number; y: number }[]) => {
    if (!onNodePositionChange) return;
    onNodePositionChange(positions.map((p) => ({ id: String(p.id), x: p.x, y: p.y })));
  };

  const scene = useGraphScene({
    nodes,
    links,
    width: graphWidth,
    height: graphHeight,
    padding,
    linkPredictionMode: !!modelType?.includes("link prediction"),
    onNodePositionChange: onGraphPositions,
  });

  return (
    <>
      <style>{styles}</style>
      <div className="dual-views-shell">
        <div className="renderer-toolbar">
          <div className="renderer-toolbar__buttons">
            {(["svg", "webgl", "webgpu", "auto"] as RendererMode[]).map((mode) => (
              <button
                key={mode}
                className={`renderer-button${
                  renderer === mode ? " renderer-button--active" : ""
                }`}
                type="button"
                onClick={() => setRenderer(mode)}
              >
                {mode.toUpperCase()}
              </button>
            ))}
          </div>
          <div className="renderer-toolbar__status">
            <span>
              Requested: <strong>{renderer.toUpperCase()}</strong>
            </span>
            <span>
              Active: <strong>{resolution.effectiveRenderer.toUpperCase()}</strong>
            </span>
            {resolution.reason ? (
              <span className="renderer-toolbar__reason">{resolution.reason}</span>
            ) : null}
          </div>
        </div>
        <div className="dual-views-grid">
          <div className="dual-views-panel">
          <GraphView
            width={graphWidth}
            height={graphHeight}
            padding={padding}
            renderer={resolution.effectiveRenderer}
            nodes={scene.nodes}
            links={scene.links}
            sceneVersion={scene.sceneVersion}
            beginDrag={scene.beginDrag}
            dragTo={scene.dragTo}
            endDrag={scene.endDrag}
            onHover={setHover}
            hover={hover}
            onRendererFailure={onBackendFailure}
          />
          </div>
          <div className="dual-views-panel">
          <MatrixView 
            width={graphWidth}
            height={graphHeight}
            padding={padding}
            renderer={resolution.effectiveRenderer}
            nodes={scene.nodes}
            links={scene.links}
            sceneVersion={scene.sceneVersion}
            onHover={setHover}
            hover={hover}
            onRendererFailure={onBackendFailure}
          />
          </div>
        </div>
      </div>
    </>
  );
};

export default DualViews;
