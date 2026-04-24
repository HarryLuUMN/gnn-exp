import type { HoverState, LinkDatum, NodeDatum } from "../dualViewTypes";

export type EngineResult<T> = T | { error: string };

export type SceneNode = NodeDatum & {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
};

export interface GraphTransform {
  translateX: number;
  translateY: number;
  scale: number;
  innerWidth: number;
  innerHeight: number;
}

export interface MatrixLayout {
  originX: number;
  originY: number;
  cellSize: number;
  size: number;
}

export interface GraphCanvasDrawArgs {
  width: number;
  height: number;
  nodes: SceneNode[];
  links: LinkDatum[];
  hover: HoverState;
  transform: GraphTransform;
}

export interface MatrixCanvasDrawArgs {
  width: number;
  height: number;
  nodes: NodeDatum[];
  matrix: number[][];
  hover: HoverState;
  layout: MatrixLayout;
}

export interface GraphCanvasEngine {
  draw: (args: GraphCanvasDrawArgs) => void;
  destroy: () => void;
}

export interface MatrixCanvasEngine {
  draw: (args: MatrixCanvasDrawArgs) => void;
  destroy: () => void;
}

export const GRAPH_NODE_RADIUS = 12;
export const GRAPH_EDGE_HIT_TOLERANCE = 10;

const NODE_FILL = "#ffffff";
const NODE_STROKE = "#69b3a2";
const NODE_HIGHLIGHT_FILL = "#006d5b";
const NODE_HIGHLIGHT_STROKE = "#004f41";
const EDGE_STROKE = "#aaaaaa";
const EDGE_HIGHLIGHT = "#006d5b";
const EDGE_AROMATIC = "#800080";
const MATRIX_ON = "#69b3a2";
const MATRIX_OFF = "#eeeeee";
const MATRIX_HIGHLIGHT = "#006d5b";

const NODE_FILL_RGBA = new Float32Array([1, 1, 1, 1]);
const NODE_STROKE_RGBA = new Float32Array([0.4118, 0.7019, 0.6353, 1]);
const NODE_HIGHLIGHT_FILL_RGBA = new Float32Array([0, 0.4275, 0.3569, 1]);
const NODE_HIGHLIGHT_STROKE_RGBA = new Float32Array([0, 0.3098, 0.2549, 1]);
const EDGE_RGBA = new Float32Array([0.6667, 0.6667, 0.6667, 1]);
const EDGE_HIGHLIGHT_RGBA = new Float32Array([0, 0.4275, 0.3569, 1]);
const EDGE_AROMATIC_RGBA = new Float32Array([0.502, 0, 0.502, 1]);
const MATRIX_ON_RGBA = new Float32Array([0.4118, 0.7019, 0.6353, 1]);
const MATRIX_OFF_RGBA = new Float32Array([0.9333, 0.9333, 0.9333, 1]);
const MATRIX_HIGHLIGHT_RGBA = new Float32Array([0, 0.4275, 0.3569, 1]);

function pushColor(target: number[], color: Float32Array) {
  target.push(color[0], color[1], color[2], color[3]);
}

export function getGraphTransform(
  width: number,
  height: number,
  padding: number,
  scaleFactor: number,
  panX: number = 0,
  panY: number = 0
): GraphTransform {
  const innerWidth = width - 2 * padding;
  const innerHeight = height - 2 * padding;

  return {
    translateX: padding + (innerWidth / 2) * (1 - scaleFactor) + panX,
    translateY: padding + (innerHeight / 2) * (1 - scaleFactor) + panY,
    scale: scaleFactor,
    innerWidth,
    innerHeight,
  };
}

export function graphPointToScreen(
  x: number,
  y: number,
  transform: GraphTransform
) {
  return {
    x: transform.translateX + x * transform.scale,
    y: transform.translateY + y * transform.scale,
  };
}

export function screenToGraphPoint(
  x: number,
  y: number,
  transform: GraphTransform
) {
  return {
    x: (x - transform.translateX) / transform.scale,
    y: (y - transform.translateY) / transform.scale,
  };
}

export function buildMatrixLayout(
  nodes: NodeDatum[],
  width: number,
  height: number,
  padding: number
): MatrixLayout {
  const size = nodes.length;
  if (size === 0) {
    return {
      originX: padding,
      originY: padding,
      cellSize: 0,
      size,
    };
  }

  const innerWidth = width - 2 * padding;
  const innerHeight = height - 2 * padding;
  const cellSize = Math.min(innerWidth / size, innerHeight / size) * 0.9;

  return {
    originX: padding,
    originY: padding,
    cellSize,
    size,
  };
}

export function buildAdjacencyMatrix(
  nodes: NodeDatum[],
  links: LinkDatum[]
): number[][] {
  const idToIndex = new Map<number, number>();
  nodes.forEach((node, index) => {
    idToIndex.set(node.id, index);
  });

  const size = nodes.length;
  const matrix: number[][] = Array.from({ length: size }, () =>
    Array(size).fill(0)
  );

  links.forEach((link) => {
    const sourceIndex = idToIndex.get(link.source);
    const targetIndex = idToIndex.get(link.target);

    if (sourceIndex == null || targetIndex == null) {
      return;
    }

    matrix[sourceIndex][targetIndex] = 1;
    matrix[targetIndex][sourceIndex] = 1;
  });

  return matrix;
}

export function getMatrixCellAtPoint(
  layout: MatrixLayout,
  x: number,
  y: number
) {
  if (layout.size === 0 || layout.cellSize <= 0) {
    return null;
  }

  const localX = x - layout.originX;
  const localY = y - layout.originY;
  const col = Math.floor(localX / layout.cellSize);
  const row = Math.floor(localY / layout.cellSize);

  if (
    row < 0 ||
    col < 0 ||
    row >= layout.size ||
    col >= layout.size
  ) {
    return null;
  }

  return { i: row, j: col };
}

export function getMatrixLabelAtPoint(
  layout: MatrixLayout,
  x: number,
  y: number
) {
  if (layout.size === 0 || layout.cellSize <= 0) {
    return null;
  }

  const topBandY = layout.originY - 18;
  const leftBandX = layout.originX - 28;

  if (y >= topBandY && y <= layout.originY) {
    const localX = x - layout.originX;
    const col = Math.floor(localX / layout.cellSize);
    if (col >= 0 && col < layout.size) {
      return { kind: "column" as const, index: col };
    }
  }

  if (x >= leftBandX && x <= layout.originX) {
    const localY = y - layout.originY;
    const row = Math.floor(localY / layout.cellSize);
    if (row >= 0 && row < layout.size) {
      return { kind: "row" as const, index: row };
    }
  }

  return null;
}

export function isNodeHighlighted(hover: HoverState, nodeId: number) {
  if (!hover) {
    return false;
  }

  if (hover.kind === "node") {
    return hover.nodeId === nodeId;
  }

  return hover.a === nodeId || hover.b === nodeId;
}

export function isLinkHighlighted(hover: HoverState, link: LinkDatum) {
  if (!hover || hover.kind !== "edge") {
    return false;
  }

  return (
    (hover.a === link.source && hover.b === link.target) ||
    (hover.a === link.target && hover.b === link.source)
  );
}

export function getNodeFillColor(hover: HoverState, nodeId: number) {
  return isNodeHighlighted(hover, nodeId) ? NODE_HIGHLIGHT_FILL : NODE_FILL;
}

export function getNodeStrokeColor(hover: HoverState, nodeId: number) {
  return isNodeHighlighted(hover, nodeId)
    ? NODE_HIGHLIGHT_STROKE
    : NODE_STROKE;
}

export function getLinkStrokeColor(hover: HoverState, link: LinkDatum) {
  if (isLinkHighlighted(hover, link)) {
    return EDGE_HIGHLIGHT;
  }

  return link.attr?.type === "aromatic" ? EDGE_AROMATIC : EDGE_STROKE;
}

export function isMatrixCellHighlighted(
  hover: HoverState,
  nodes: NodeDatum[],
  row: number,
  col: number
) {
  if (!hover) {
    return false;
  }

  const rowId = nodes[row]?.id;
  const colId = nodes[col]?.id;

  if (rowId == null || colId == null) {
    return false;
  }

  if (hover.kind === "node") {
    return rowId === hover.nodeId || colId === hover.nodeId;
  }

  return (
    (rowId === hover.a && colId === hover.b) ||
    (rowId === hover.b && colId === hover.a)
  );
}

export function getMatrixCellColor(
  hover: HoverState,
  nodes: NodeDatum[],
  row: number,
  col: number,
  value: number
) {
  if (isMatrixCellHighlighted(hover, nodes, row, col)) {
    return MATRIX_HIGHLIGHT;
  }

  return value ? MATRIX_ON : MATRIX_OFF;
}

function getNodeFillColorVector(hover: HoverState, nodeId: number) {
  return isNodeHighlighted(hover, nodeId)
    ? NODE_HIGHLIGHT_FILL_RGBA
    : NODE_FILL_RGBA;
}

function getNodeStrokeColorVector(hover: HoverState, nodeId: number) {
  return isNodeHighlighted(hover, nodeId)
    ? NODE_HIGHLIGHT_STROKE_RGBA
    : NODE_STROKE_RGBA;
}

function getLinkColorVector(hover: HoverState, link: LinkDatum) {
  if (isLinkHighlighted(hover, link)) {
    return EDGE_HIGHLIGHT_RGBA;
  }

  return link.attr?.type === "aromatic" ? EDGE_AROMATIC_RGBA : EDGE_RGBA;
}

function getMatrixColorVector(
  hover: HoverState,
  nodes: NodeDatum[],
  row: number,
  col: number,
  value: number
) {
  if (isMatrixCellHighlighted(hover, nodes, row, col)) {
    return MATRIX_HIGHLIGHT_RGBA;
  }

  return value ? MATRIX_ON_RGBA : MATRIX_OFF_RGBA;
}

export function findNodeAtPoint(
  nodes: SceneNode[],
  x: number,
  y: number,
  radius: number = GRAPH_NODE_RADIUS
) {
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const node = nodes[index];
    const dx = (node.x ?? 0) - x;
    const dy = (node.y ?? 0) - y;
    if (dx * dx + dy * dy <= radius * radius) {
      return node;
    }
  }

  return null;
}

function distanceToSegment(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
) {
  const dx = x2 - x1;
  const dy = y2 - y1;

  if (dx === 0 && dy === 0) {
    return Math.hypot(px - x1, py - y1);
  }

  const t = Math.max(
    0,
    Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy))
  );
  const cx = x1 + t * dx;
  const cy = y1 + t * dy;

  return Math.hypot(px - cx, py - cy);
}

export function findEdgeAtPoint(
  nodesById: Map<number, SceneNode>,
  links: LinkDatum[],
  x: number,
  y: number,
  tolerance: number = GRAPH_EDGE_HIT_TOLERANCE
): LinkDatum | null {
  let closest: LinkDatum | null = null;
  let closestDistance = tolerance;

  links.forEach((link) => {
    const source = nodesById.get(link.source);
    const target = nodesById.get(link.target);

    if (!source || !target) {
      return;
    }

    const distance = distanceToSegment(
      x,
      y,
      source.x ?? 0,
      source.y ?? 0,
      target.x ?? 0,
      target.y ?? 0
    );

    if (distance <= closestDistance) {
      closest = link;
      closestDistance = distance;
    }
  });

  return closest;
}

export function buildGraphGeometry(
  nodes: SceneNode[],
  links: LinkDatum[],
  nodesById: Map<number, SceneNode>,
  hover: HoverState,
  transform: GraphTransform
) {
  const edgeVertices: number[] = [];
  const nodeVertices: number[] = [];
  const nodeRadius = GRAPH_NODE_RADIUS * transform.scale;

  links.forEach((link) => {
    const source = nodesById.get(link.source);
    const target = nodesById.get(link.target);

    if (!source || !target) {
      return;
    }

    const sourceScreen = graphPointToScreen(source.x ?? 0, source.y ?? 0, transform);
    const targetScreen = graphPointToScreen(target.x ?? 0, target.y ?? 0, transform);
    const color = getLinkColorVector(hover, link);

    edgeVertices.push(sourceScreen.x, sourceScreen.y);
    pushColor(edgeVertices, color);
    edgeVertices.push(targetScreen.x, targetScreen.y);
    pushColor(edgeVertices, color);
  });

  nodes.forEach((node) => {
    const center = graphPointToScreen(node.x ?? 0, node.y ?? 0, transform);
    const fill = getNodeFillColorVector(hover, node.id);
    const stroke = getNodeStrokeColorVector(hover, node.id);
    const quad = [
      [-nodeRadius, -nodeRadius, -1, -1],
      [nodeRadius, -nodeRadius, 1, -1],
      [nodeRadius, nodeRadius, 1, 1],
      [-nodeRadius, -nodeRadius, -1, -1],
      [nodeRadius, nodeRadius, 1, 1],
      [-nodeRadius, nodeRadius, -1, 1],
    ] as const;

    quad.forEach(([dx, dy, localX, localY]) => {
      nodeVertices.push(center.x + dx, center.y + dy, localX, localY);
      pushColor(nodeVertices, fill);
      pushColor(nodeVertices, stroke);
    });
  });

  return {
    edgeVertices: new Float32Array(edgeVertices),
    nodeVertices: new Float32Array(nodeVertices),
  };
}

export function buildMatrixGeometry(
  nodes: NodeDatum[],
  matrix: number[][],
  hover: HoverState,
  layout: MatrixLayout
) {
  const vertices: number[] = [];

  if (layout.size === 0 || layout.cellSize <= 0) {
    return new Float32Array(vertices);
  }

  const inset = Math.min(1, layout.cellSize * 0.08);
  const rectSize = Math.max(layout.cellSize - inset * 2, 0);

  for (let row = 0; row < layout.size; row += 1) {
    for (let col = 0; col < layout.size; col += 1) {
      const x = layout.originX + col * layout.cellSize + inset;
      const y = layout.originY + row * layout.cellSize + inset;
      const color = getMatrixColorVector(
        hover,
        nodes,
        row,
        col,
        matrix[row]?.[col] ?? 0
      );
      const rect = [
        [x, y],
        [x + rectSize, y],
        [x + rectSize, y + rectSize],
        [x, y],
        [x + rectSize, y + rectSize],
        [x, y + rectSize],
      ] as const;

      rect.forEach(([vx, vy]) => {
        vertices.push(vx, vy);
        pushColor(vertices, color);
      });
    }
  }

  return new Float32Array(vertices);
}
