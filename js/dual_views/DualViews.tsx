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

const BASE_GRAPH_SCALE = 1.4;
const MIN_GRAPH_ZOOM = 0.5;
const MAX_GRAPH_ZOOM = 4;
const GRAPH_ZOOM_STEP = 0.25;
const MATRIX_AXIS_LABEL_LIMIT = 80;
const VIEW_WIDTH = 1200;
const VIEW_HEIGHT = 900;

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
  const [graphZoom, setGraphZoom] = useState(1);
  const [graphPan, setGraphPan] = useState({ x: 0, y: 0 });
  const [fallbackFailures, setFallbackFailures] = useState<
    Partial<Record<ResolvedRenderer, string>>
  >({});

  const styles = dualViewVisualizerStyle;
  const graphWidth = VIEW_WIDTH;
  const graphHeight = VIEW_HEIGHT;
  const padding = 60;
  const capabilities = React.useMemo(() => detectCapabilities(), []);
  const resolution = React.useMemo(
    () => resolveRenderer(renderer, capabilities, fallbackFailures),
    [capabilities, fallbackFailures, renderer]
  );
  const graphScaleFactor = BASE_GRAPH_SCALE * graphZoom;
  const showMatrixAxisLabels = nodes.length <= MATRIX_AXIS_LABEL_LIMIT;
  const isGraphViewReset =
    graphZoom === 1 && graphPan.x === 0 && graphPan.y === 0;

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

  const zoomOut = React.useCallback(() => {
    setGraphZoom((value) => Math.max(MIN_GRAPH_ZOOM, value - GRAPH_ZOOM_STEP));
  }, []);

  const zoomIn = React.useCallback(() => {
    setGraphZoom((value) => Math.min(MAX_GRAPH_ZOOM, value + GRAPH_ZOOM_STEP));
  }, []);

  const resetZoom = React.useCallback(() => {
    setGraphZoom(1);
    setGraphPan({ x: 0, y: 0 });
  }, []);

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
            {!showMatrixAxisLabels ? (
              <span className="renderer-toolbar__reason">
                Matrix axis labels hidden for {nodes.length} nodes.
              </span>
            ) : null}
          </div>
          <div className="renderer-toolbar__zoom">
            <span>Graph zoom: {Math.round(graphZoom * 100)}%</span>
            <span className="renderer-toolbar__hint">Drag empty graph space to pan.</span>
            <button
              className="renderer-button"
              type="button"
              onClick={zoomOut}
              disabled={graphZoom <= MIN_GRAPH_ZOOM}
            >
              Zoom Out
            </button>
            <button
              className="renderer-button"
              type="button"
              onClick={resetZoom}
              disabled={isGraphViewReset}
            >
              Reset
            </button>
            <button
              className="renderer-button"
              type="button"
              onClick={zoomIn}
              disabled={graphZoom >= MAX_GRAPH_ZOOM}
            >
              Zoom In
            </button>
          </div>
        </div>
        <div className="dual-views-grid">
          <div className="dual-views-panel">
          <GraphView
            width={graphWidth}
            height={graphHeight}
            padding={padding}
            scaleFactor={graphScaleFactor}
            renderer={resolution.effectiveRenderer}
            nodes={scene.nodes}
            links={scene.links}
            sceneVersion={scene.sceneVersion}
            beginDrag={scene.beginDrag}
            dragTo={scene.dragTo}
            endDrag={scene.endDrag}
            onHover={setHover}
            hover={hover}
            panOffset={graphPan}
            onPanChange={setGraphPan}
            onRendererFailure={onBackendFailure}
          />
          </div>
          <div className="dual-views-panel">
          <MatrixView 
            width={graphWidth}
            height={graphHeight}
            padding={padding}
            renderer={resolution.effectiveRenderer}
            showAxisLabels={showMatrixAxisLabels}
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
