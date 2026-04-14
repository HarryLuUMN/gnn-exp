import React from "react";
import type { HoverState, LinkDatum } from "./dualViewTypes";
import type { ResolvedRenderer } from "./renderers/capabilities";
import {
  findEdgeAtPoint,
  findNodeAtPoint,
  getGraphTransform,
  getLinkStrokeColor,
  getNodeFillColor,
  getNodeStrokeColor,
  isLinkHighlighted,
  isNodeHighlighted,
  screenToGraphPoint,
  type GraphCanvasDrawArgs,
  type GraphCanvasEngine,
  type SceneNode,
} from "./renderers/shared";
import { createWebglGraphEngine } from "./renderers/webgl";
import { createWebgpuGraphEngine } from "./renderers/webgpu";

interface GraphViewProps {
  width?: number;
  height?: number;
  padding?: number;
  scaleFactor?: number;
  renderer: ResolvedRenderer;
  nodes: SceneNode[];
  links: LinkDatum[];
  sceneVersion: number;
  beginDrag: (nodeId: number) => boolean;
  dragTo: (x: number, y: number) => void;
  endDrag: () => void;
  onHover?: (h: HoverState) => void;
  hover?: HoverState;
  onRendererFailure: (
    backend: Exclude<ResolvedRenderer, "svg">,
    reason: string
  ) => void;
}

interface CanvasGraphLayerProps {
  backend: Exclude<ResolvedRenderer, "svg">;
  drawArgs: GraphCanvasDrawArgs;
  sceneVersion: number;
  onRendererFailure: (
    backend: Exclude<ResolvedRenderer, "svg">,
    reason: string
  ) => void;
}

const CanvasGraphLayer: React.FC<CanvasGraphLayerProps> = ({
  backend,
  drawArgs,
  sceneVersion,
  onRendererFailure,
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const engineRef = React.useRef<GraphCanvasEngine | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    engineRef.current?.destroy();
    engineRef.current = null;

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    if (backend === "webgl") {
      const result = createWebglGraphEngine(canvas);
      if ("error" in result) {
        onRendererFailure("webgl", result.error);
        return;
      }

      engineRef.current = result;
      result.draw(drawArgs);
      return () => {
        result.destroy();
        engineRef.current = null;
      };
    }

    createWebgpuGraphEngine(canvas).then((result) => {
      if (cancelled) {
        if (!("error" in result)) {
          result.destroy();
        }
        return;
      }

      if ("error" in result) {
        onRendererFailure("webgpu", result.error);
        return;
      }

      engineRef.current = result;
      result.draw(drawArgs);
    });

    return () => {
      cancelled = true;
      engineRef.current?.destroy();
      engineRef.current = null;
    };
  }, [backend, onRendererFailure]);

  React.useEffect(() => {
    engineRef.current?.draw(drawArgs);
  }, [drawArgs, sceneVersion]);

  return (
    <canvas
      ref={canvasRef}
      className="dual-views-canvas"
      style={{ width: drawArgs.width, height: drawArgs.height }}
    />
  );
};

const SvgGraphRenderer: React.FC<{
  width: number;
  height: number;
  transform: ReturnType<typeof getGraphTransform>;
  nodes: SceneNode[];
  links: LinkDatum[];
  hover?: HoverState;
}> = ({ width, height, transform, nodes, links, hover }) => {
  const nodesById = React.useMemo(
    () => new Map(nodes.map((node) => [node.id, node] as const)),
    [nodes]
  );

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <g
        transform={`translate(${transform.translateX},${transform.translateY}) scale(${transform.scale})`}
      >
        {links.map((link, index) => {
          const source = nodesById.get(link.source);
          const target = nodesById.get(link.target);
          if (!source || !target) {
            return null;
          }

          return (
            <line
              key={`${link.source}-${link.target}-${index}`}
              className="link"
              x1={source.x ?? 0}
              y1={source.y ?? 0}
              x2={target.x ?? 0}
              y2={target.y ?? 0}
              stroke={getLinkStrokeColor(hover ?? null, link)}
              strokeWidth={isLinkHighlighted(hover ?? null, link) ? 2 : 1}
              strokeOpacity={isLinkHighlighted(hover ?? null, link) ? 1 : 0.6}
            />
          );
        })}
        {nodes.map((node) => (
          <g key={node.id} transform={`translate(${node.x ?? 0},${node.y ?? 0})`}>
            <circle
              className="node"
              r={12}
              fill={getNodeFillColor(hover ?? null, node.id)}
              stroke={getNodeStrokeColor(hover ?? null, node.id)}
              strokeWidth={isNodeHighlighted(hover ?? null, node.id) ? 2 : 1.5}
            />
            <text className="node-label" dy={4}>
              {node.element}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
};

const GraphView: React.FC<GraphViewProps> = ({
  width = 800,
  height = 600,
  padding = 60,
  scaleFactor = 1.4,
  renderer,
  nodes,
  links,
  sceneVersion,
  beginDrag,
  dragTo,
  endDrag,
  onHover,
  hover,
  onRendererFailure,
}) => {
  const transform = React.useMemo(
    () => getGraphTransform(width, height, padding, scaleFactor),
    [height, padding, scaleFactor, width]
  );
  const nodesById = React.useMemo(
    () => new Map(nodes.map((node) => [node.id, node] as const)),
    [nodes, sceneVersion]
  );
  const draggingRef = React.useRef<number | null>(null);

  const drawArgs = React.useMemo<GraphCanvasDrawArgs>(
    () => ({
      width,
      height,
      nodes,
      links,
      hover: hover ?? null,
      transform,
    }),
    [height, hover, links, nodes, transform, width]
  );

  const updateHoverFromPoint = React.useCallback(
    (screenX: number, screenY: number) => {
      const graphPoint = screenToGraphPoint(screenX, screenY, transform);
      const hitNode = findNodeAtPoint(nodes, graphPoint.x, graphPoint.y);
      if (hitNode) {
        onHover?.({ kind: "node", nodeId: hitNode.id });
        return;
      }

      const hitEdge = findEdgeAtPoint(nodesById, links, graphPoint.x, graphPoint.y);
      if (hitEdge) {
        onHover?.({ kind: "edge", a: hitEdge.source, b: hitEdge.target });
        return;
      }

      onHover?.(null);
    },
    [links, nodes, nodesById, onHover, transform]
  );

  const getPointerPosition = React.useCallback(
    (event: React.PointerEvent<SVGRectElement>) => {
      const bounds = event.currentTarget.getBoundingClientRect();
      return {
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      };
    },
    []
  );

  const handlePointerDown = React.useCallback(
    (event: React.PointerEvent<SVGRectElement>) => {
      const { x, y } = getPointerPosition(event);
      const graphPoint = screenToGraphPoint(x, y, transform);
      const hitNode = findNodeAtPoint(nodes, graphPoint.x, graphPoint.y);

      if (!hitNode) {
        return;
      }

      if (!beginDrag(hitNode.id)) {
        return;
      }

      draggingRef.current = hitNode.id;
      onHover?.({ kind: "node", nodeId: hitNode.id });
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [beginDrag, getPointerPosition, nodes, onHover, transform]
  );

  const handlePointerMove = React.useCallback(
    (event: React.PointerEvent<SVGRectElement>) => {
      const { x, y } = getPointerPosition(event);

      if (draggingRef.current !== null) {
        const graphPoint = screenToGraphPoint(x, y, transform);
        dragTo(graphPoint.x, graphPoint.y);
        onHover?.({ kind: "node", nodeId: draggingRef.current });
        return;
      }

      updateHoverFromPoint(x, y);
    },
    [dragTo, getPointerPosition, onHover, transform, updateHoverFromPoint]
  );

  const handlePointerUp = React.useCallback(
    (event: React.PointerEvent<SVGRectElement>) => {
      const { x, y } = getPointerPosition(event);
      if (draggingRef.current !== null) {
        draggingRef.current = null;
        endDrag();
      }
      updateHoverFromPoint(x, y);
    },
    [endDrag, getPointerPosition, updateHoverFromPoint]
  );

  const handlePointerLeave = React.useCallback(() => {
    if (draggingRef.current !== null) {
      draggingRef.current = null;
      endDrag();
    }

    onHover?.(null);
  }, [endDrag, onHover]);

  const overlayLabels = (
    <g
      pointerEvents="none"
      transform={`translate(${transform.translateX},${transform.translateY}) scale(${transform.scale})`}
    >
      {nodes.map((node) => (
        <text
          key={node.id}
          className="node-label"
          x={node.x ?? 0}
          y={(node.y ?? 0) + 4}
        >
          {node.element}
        </text>
      ))}
    </g>
  );

  if (renderer === "svg") {
    return (
      <div className="dual-views-stage" style={{ width, height }}>
        <SvgGraphRenderer
          width={width}
          height={height}
          transform={transform}
          nodes={nodes}
          links={links}
          hover={hover}
        />
        <svg className="dual-views-overlay" width={width} height={height}>
          <rect
            width={width}
            height={height}
            fill="transparent"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerLeave}
            onPointerCancel={handlePointerLeave}
          />
        </svg>
      </div>
    );
  }

  return (
    <div className="dual-views-stage" style={{ width, height }}>
      <CanvasGraphLayer
        backend={renderer}
        drawArgs={drawArgs}
        sceneVersion={sceneVersion}
        onRendererFailure={onRendererFailure}
      />
      <svg className="dual-views-overlay" width={width} height={height}>
        {overlayLabels}
        <rect
          width={width}
          height={height}
          fill="transparent"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerLeave}
          onPointerCancel={handlePointerLeave}
        />
      </svg>
    </div>
  );
};

export default GraphView;
