import React from "react";
import type { HoverState, LinkDatum, NodeDatum } from "./dualViewTypes";
import type { ResolvedRenderer } from "./renderers/capabilities";
import {
  buildAdjacencyMatrix,
  buildMatrixLayout,
  getMatrixCellAtPoint,
  getMatrixCellColor,
  getMatrixLabelAtPoint,
  isMatrixCellHighlighted,
  type MatrixCanvasDrawArgs,
  type MatrixCanvasEngine,
} from "./renderers/shared";
import { createWebglMatrixEngine } from "./renderers/webgl";
import { createWebgpuMatrixEngine } from "./renderers/webgpu";

interface MatrixViewProps {
  width?: number;
  height?: number;
  padding?: number;
  renderer: ResolvedRenderer;
  showAxisLabels?: boolean;
  nodes: NodeDatum[];
  links: LinkDatum[];
  sceneVersion: number;
  onHover?: (h: HoverState) => void;
  hover?: HoverState;
  onRendererFailure: (
    backend: Exclude<ResolvedRenderer, "svg">,
    reason: string
  ) => void;
}

interface CanvasMatrixLayerProps {
  backend: Exclude<ResolvedRenderer, "svg">;
  drawArgs: MatrixCanvasDrawArgs;
  sceneVersion: number;
  onRendererFailure: (
    backend: Exclude<ResolvedRenderer, "svg">,
    reason: string
  ) => void;
}

const CanvasMatrixLayer: React.FC<CanvasMatrixLayerProps> = ({
  backend,
  drawArgs,
  sceneVersion,
  onRendererFailure,
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const engineRef = React.useRef<MatrixCanvasEngine | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    engineRef.current?.destroy();
    engineRef.current = null;

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    if (backend === "webgl") {
      const result = createWebglMatrixEngine(canvas);
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

    createWebgpuMatrixEngine(canvas).then((result) => {
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
      className="dual-views-layer dual-views-canvas"
      style={{ width: "100%", height: "100%" }}
    />
  );
};

const SvgMatrixRenderer: React.FC<{
  width: number;
  height: number;
  nodes: NodeDatum[];
  matrix: number[][];
  layout: ReturnType<typeof buildMatrixLayout>;
  hover?: HoverState;
  showAxisLabels: boolean;
}> = ({ width, height, nodes, matrix, layout, hover, showAxisLabels }) => {
  return (
    <svg
      className="dual-views-layer"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      <g>
        {matrix.map((row, rowIndex) =>
          row.map((value, colIndex) => (
            <rect
              key={`${rowIndex}-${colIndex}`}
              className="matrix-cell"
              x={layout.originX + colIndex * layout.cellSize}
              y={layout.originY + rowIndex * layout.cellSize}
              width={layout.cellSize}
              height={layout.cellSize}
              fill={getMatrixCellColor(
                hover ?? null,
                nodes,
                rowIndex,
                colIndex,
                value
              )}
              stroke={isMatrixCellHighlighted(hover ?? null, nodes, rowIndex, colIndex) ? "#004f41" : "#ffffff"}
              strokeWidth={isMatrixCellHighlighted(hover ?? null, nodes, rowIndex, colIndex) ? 1 : 0.5}
            />
          ))
        )}
        {showAxisLabels ? (
          <>
            {nodes.map((node, index) => (
              <text
                key={`top-${node.id}`}
                className="axis-label"
                x={layout.originX + index * layout.cellSize + layout.cellSize / 2}
                y={layout.originY - 6}
                textAnchor="middle"
              >
                {node.id}
              </text>
            ))}
            {nodes.map((node, index) => (
              <text
                key={`left-${node.id}`}
                className="axis-label"
                x={layout.originX - 6}
                y={layout.originY + index * layout.cellSize + layout.cellSize / 2}
                textAnchor="end"
                dominantBaseline="middle"
              >
                {node.id}
              </text>
            ))}
          </>
        ) : null}
      </g>
    </svg>
  );
};

const MatrixView: React.FC<MatrixViewProps> = ({
  width = 800,
  height = 600,
  padding = 60,
  renderer,
  showAxisLabels = true,
  nodes,
  links,
  sceneVersion,
  onHover,
  hover,
  onRendererFailure,
}) => {
  const matrix = React.useMemo(() => buildAdjacencyMatrix(nodes, links), [links, nodes]);
  const layout = React.useMemo(
    () => buildMatrixLayout(nodes, width, height, padding),
    [height, nodes, padding, width]
  );

  const drawArgs = React.useMemo<MatrixCanvasDrawArgs>(
    () => ({
      width,
      height,
      nodes,
      matrix,
      hover: hover ?? null,
      layout,
    }),
    [height, hover, layout, matrix, nodes, width]
  );

  const getPointerPosition = React.useCallback(
    (event: React.PointerEvent<SVGRectElement>) => {
      const bounds = event.currentTarget.getBoundingClientRect();
      const scaleX = bounds.width === 0 ? 1 : width / bounds.width;
      const scaleY = bounds.height === 0 ? 1 : height / bounds.height;
      return {
        x: (event.clientX - bounds.left) * scaleX,
        y: (event.clientY - bounds.top) * scaleY,
      };
    },
    [height, width]
  );

  const updateHoverFromPoint = React.useCallback(
    (x: number, y: number) => {
      if (showAxisLabels) {
        const labelHit = getMatrixLabelAtPoint(layout, x, y);
        if (labelHit) {
          onHover?.({
            kind: "node",
            nodeId: nodes[labelHit.index]?.id ?? -1,
          });
          return;
        }
      }

      const cell = getMatrixCellAtPoint(layout, x, y);
      if (!cell) {
        onHover?.(null);
        return;
      }

      if ((matrix[cell.i]?.[cell.j] ?? 0) === 0) {
        onHover?.(null);
        return;
      }

      onHover?.({
        kind: "edge",
        a: nodes[cell.i]?.id ?? -1,
        b: nodes[cell.j]?.id ?? -1,
      });
    },
    [layout, matrix, nodes, onHover, showAxisLabels]
  );

  const handlePointerMove = React.useCallback(
    (event: React.PointerEvent<SVGRectElement>) => {
      const { x, y } = getPointerPosition(event);
      updateHoverFromPoint(x, y);
    },
    [getPointerPosition, updateHoverFromPoint]
  );

  const handlePointerLeave = React.useCallback(() => {
    onHover?.(null);
  }, [onHover]);

  const overlay = (
    <svg
      className="dual-views-layer dual-views-overlay"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      <rect
        width={width}
        height={height}
        fill="transparent"
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onPointerCancel={handlePointerLeave}
      />
    </svg>
  );

  if (renderer === "svg") {
    return (
      <div className="dual-views-stage" style={{ width, height }}>
        <SvgMatrixRenderer
          width={width}
          height={height}
          nodes={nodes}
          matrix={matrix}
          layout={layout}
          hover={hover}
          showAxisLabels={showAxisLabels}
        />
        {overlay}
      </div>
    );
  }

  return (
    <div className="dual-views-stage" style={{ width, height }}>
      <CanvasMatrixLayer
        backend={renderer}
        drawArgs={drawArgs}
        sceneVersion={sceneVersion}
        onRendererFailure={onRendererFailure}
      />
      <svg
        className="dual-views-layer dual-views-overlay"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
      >
        {showAxisLabels ? (
          <g pointerEvents="none">
            {nodes.map((node, index) => (
              <text
                key={`top-${node.id}`}
                className="axis-label"
                x={layout.originX + index * layout.cellSize + layout.cellSize / 2}
                y={layout.originY - 6}
                textAnchor="middle"
              >
                {node.id}
              </text>
            ))}
            {nodes.map((node, index) => (
              <text
                key={`left-${node.id}`}
                className="axis-label"
                x={layout.originX - 6}
                y={layout.originY + index * layout.cellSize + layout.cellSize / 2}
                textAnchor="end"
                dominantBaseline="middle"
              >
                {node.id}
              </text>
            ))}
          </g>
        ) : null}
        <rect
          width={width}
          height={height}
          fill="transparent"
          onPointerMove={handlePointerMove}
          onPointerLeave={handlePointerLeave}
          onPointerCancel={handlePointerLeave}
        />
      </svg>
    </div>
  );
};

export default MatrixView;
