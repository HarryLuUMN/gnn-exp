import { featureColor } from "../utils/const";
import {
  extractSortedGNNLayerFeatures,
  processSubgraphSequenceDataPipe,
  type SubgraphResult,
} from "../utils/dataProcessingUtils";
import { addVector, divideVector, randomVector } from "../utils/mathUtils";

export type Rgba = [number, number, number, number];

export type StaticPrimitive =
  | {
      kind: "rect";
      x: number;
      y: number;
      width: number;
      height: number;
      fill?: Rgba;
      stroke?: Rgba;
      strokeWidth?: number;
    }
  | {
      kind: "polyline";
      points: Array<[number, number]>;
      stroke: Rgba;
      strokeWidth: number;
    };

export interface StaticVisualizationScene {
  width: number;
  height: number;
  primitives: StaticPrimitive[];
}

const MATRIX_START_X = 50;
const MATRIX_START_Y = 50;
const MATRIX_CELL_SIZE = 20;
const NODE_ROW_HEIGHT = 20;
const LAYER_GAP = 100;
const SCENE_PADDING = 24;
const MATRIX_ON: Rgba = [105 / 255, 179 / 255, 162 / 255, 0.8];
const MATRIX_OFF: Rgba = [238 / 255, 238 / 255, 238 / 255, 0.8];
const WHITE: Rgba = [1, 1, 1, 1];
const BLACK_FAINT: Rgba = [0, 0, 0, 0.1];
const BLACK_HALF: Rgba = [0, 0, 0, 0.5];
const GRAY_HALF: Rgba = [0.5, 0.5, 0.5, 0.5];

type SceneBuilder = {
  primitives: StaticPrimitive[];
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

function createBuilder(): SceneBuilder {
  return {
    primitives: [],
    minX: Number.POSITIVE_INFINITY,
    minY: Number.POSITIVE_INFINITY,
    maxX: Number.NEGATIVE_INFINITY,
    maxY: Number.NEGATIVE_INFINITY,
  };
}

function includeBounds(
  builder: SceneBuilder,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number
) {
  builder.minX = Math.min(builder.minX, minX);
  builder.minY = Math.min(builder.minY, minY);
  builder.maxX = Math.max(builder.maxX, maxX);
  builder.maxY = Math.max(builder.maxY, maxY);
}

function addRect(
  builder: SceneBuilder,
  x: number,
  y: number,
  width: number,
  height: number,
  options: {
    fill?: Rgba;
    stroke?: Rgba;
    strokeWidth?: number;
  }
) {
  const strokeWidth = options.strokeWidth ?? 0;
  builder.primitives.push({
    kind: "rect",
    x,
    y,
    width,
    height,
    fill: options.fill,
    stroke: options.stroke,
    strokeWidth,
  });
  includeBounds(
    builder,
    x - strokeWidth / 2,
    y - strokeWidth / 2,
    x + width + strokeWidth / 2,
    y + height + strokeWidth / 2
  );
}

function addPolyline(
  builder: SceneBuilder,
  points: Array<[number, number]>,
  stroke: Rgba,
  strokeWidth: number
) {
  if (points.length < 2) {
    return;
  }

  builder.primitives.push({ kind: "polyline", points, stroke, strokeWidth });
  for (const [x, y] of points) {
    includeBounds(
      builder,
      x - strokeWidth / 2,
      y - strokeWidth / 2,
      x + strokeWidth / 2,
      y + strokeWidth / 2
    );
  }
}

function parseColor(color: string, alpha: number = 1): Rgba {
  if (color === "white") {
    return [1, 1, 1, alpha];
  }
  if (color === "black") {
    return [0, 0, 0, alpha];
  }
  if (color === "gray" || color === "grey") {
    return [0.5, 0.5, 0.5, alpha];
  }

  const rgb = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (rgb) {
    return [
      Number(rgb[1]) / 255,
      Number(rgb[2]) / 255,
      Number(rgb[3]) / 255,
      alpha,
    ];
  }

  const hex = color.match(/^#([0-9a-f]{6})$/i);
  if (hex) {
    const value = Number.parseInt(hex[1], 16);
    return [
      ((value >> 16) & 255) / 255,
      ((value >> 8) & 255) / 255,
      (value & 255) / 255,
      alpha,
    ];
  }

  return [0, 0, 0, alpha];
}

function featureRgba(value: number): Rgba {
  return parseColor(featureColor(value), 1);
}

function sampleCubic(
  start: [number, number],
  controlA: [number, number],
  controlB: [number, number],
  end: [number, number],
  segments: number = 16
): Array<[number, number]> {
  const points: Array<[number, number]> = [];
  for (let i = 0; i <= segments; i += 1) {
    const t = i / segments;
    const mt = 1 - t;
    const x =
      mt * mt * mt * start[0] +
      3 * mt * mt * t * controlA[0] +
      3 * mt * t * t * controlB[0] +
      t * t * t * end[0];
    const y =
      mt * mt * mt * start[1] +
      3 * mt * mt * t * controlA[1] +
      3 * mt * t * t * controlB[1] +
      t * t * t * end[1];
    points.push([x, y]);
  }
  return points;
}

function shouldRenderNode(
  subgraphSample: boolean,
  subgraph: SubgraphResult | undefined,
  nodeIndex: number
) {
  return !subgraphSample || !!subgraph?.nodes.includes(nodeIndex);
}

function addFeatureVector(
  builder: SceneBuilder,
  x: number,
  y: number,
  feature: number[],
  cellWidth: number,
  cellHeight: number,
  frameStrokeWidth: number
) {
  addRect(builder, x, y, feature.length * cellWidth, cellHeight, {
    stroke: frameStrokeWidth > 1 ? BLACK_HALF : BLACK_HALF,
    strokeWidth: frameStrokeWidth,
  });

  for (let index = 0; index < feature.length; index += 1) {
    addRect(builder, x + index * cellWidth, y, cellWidth, cellHeight, {
      fill: featureRgba(feature[index]),
      stroke: GRAY_HALF,
      strokeWidth: 0.5,
    });
  }
}

function addMatrix(builder: SceneBuilder, adjacencyMatrix: number[][]) {
  const size = adjacencyMatrix.length;
  const matrixSize = size * MATRIX_CELL_SIZE;
  addRect(builder, MATRIX_START_X, MATRIX_START_Y, matrixSize, matrixSize, {
    fill: MATRIX_OFF,
  });

  for (let row = 0; row < adjacencyMatrix.length; row += 1) {
    for (let col = 0; col < adjacencyMatrix[row].length; col += 1) {
      if (adjacencyMatrix[row][col] !== 1) {
        continue;
      }

      addRect(
        builder,
        MATRIX_START_X + col * MATRIX_CELL_SIZE,
        MATRIX_START_Y + row * MATRIX_CELL_SIZE,
        MATRIX_CELL_SIZE,
        MATRIX_CELL_SIZE,
        {
          fill: MATRIX_ON,
          stroke: WHITE,
          strokeWidth: 1,
        }
      );
    }
  }
}

function addIntermediateFeatures(
  builder: SceneBuilder,
  sortedGNNFeatures: number[][][],
  cellWidth: number,
  cellHeight: number,
  adjacencyMatrix: number[][],
  subgraphData: SubgraphResult[],
  subgraphSample: boolean
) {
  const startX =
    adjacencyMatrix.length * MATRIX_CELL_SIZE + MATRIX_CELL_SIZE + MATRIX_START_X;
  let layerX = startX;

  for (let layerIndex = 0; layerIndex < sortedGNNFeatures.length; layerIndex += 1) {
    const previousFeatureDim = sortedGNNFeatures[layerIndex - 1]?.[0]?.length ?? 0;
    if (layerIndex > 0) {
      layerX += cellWidth * previousFeatureDim;
    }

    const subgraph = subgraphData[layerIndex];
    const layerFeatures = sortedGNNFeatures[layerIndex] ?? [];
    for (let nodeIndex = 0; nodeIndex < layerFeatures.length; nodeIndex += 1) {
      if (!shouldRenderNode(subgraphSample, subgraph, nodeIndex)) {
        continue;
      }

      addFeatureVector(
        builder,
        layerX,
        MATRIX_START_Y + nodeIndex * NODE_ROW_HEIGHT + cellHeight / 2,
        layerFeatures[nodeIndex],
        cellWidth,
        cellHeight,
        2
      );
    }

    layerX += LAYER_GAP;
  }

  return layerX;
}

function addBetweenLayerLinks(
  builder: SceneBuilder,
  sortedGNNFeatures: number[][][],
  links: any[],
  cellWidth: number,
  subgraphData: SubgraphResult[],
  subgraphSample: boolean
) {
  const startX =
    MATRIX_START_X +
    sortedGNNFeatures[0].length * MATRIX_CELL_SIZE +
    MATRIX_CELL_SIZE;
  let layerX = startX + sortedGNNFeatures[0][0].length * cellWidth;

  for (let layerIndex = 0; layerIndex < sortedGNNFeatures.length - 1; layerIndex += 1) {
    const previousLayerX = layerX;
    layerX += sortedGNNFeatures[layerIndex + 1][0].length * cellWidth + LAYER_GAP;
    const midLayerX = (previousLayerX + layerX) / 2;
    const subgraph = subgraphData[layerIndex + 1];

    for (const link of links) {
      const sourceIndex = link.source;
      const targetIndex = link.target;
      if (!shouldRenderNode(subgraphSample, subgraph, targetIndex)) {
        continue;
      }

      const sourceY = MATRIX_START_Y + sourceIndex * NODE_ROW_HEIGHT + 12;
      const targetY = MATRIX_START_Y + targetIndex * NODE_ROW_HEIGHT + 12;
      addPolyline(
        builder,
        sampleCubic(
          [previousLayerX, sourceY],
          [midLayerX, sourceY],
          [midLayerX, targetY],
          [layerX, targetY]
        ),
        BLACK_FAINT,
        1
      );
    }

    for (let nodeIndex = 0; nodeIndex < sortedGNNFeatures[layerIndex].length; nodeIndex += 1) {
      if (!shouldRenderNode(subgraphSample, subgraph, nodeIndex)) {
        continue;
      }

      const layerY = MATRIX_START_Y + nodeIndex * NODE_ROW_HEIGHT + 12;
      addPolyline(builder, [[layerX, layerY], [layerX - 125, layerY]], BLACK_FAINT, 1);
    }
  }
}

function addSingleFC(
  builder: SceneBuilder,
  layerX: number,
  layerY: number,
  feature: number[],
  nodeIndex: number,
  cellWidth: number
) {
  const y = layerY + nodeIndex * NODE_ROW_HEIGHT + 6;
  addFeatureVector(builder, layerX, y, feature, cellWidth, 12, 2);
  addPolyline(
    builder,
    [[layerX, layerY + nodeIndex * NODE_ROW_HEIGHT + 12], [layerX - LAYER_GAP, layerY + nodeIndex * NODE_ROW_HEIGHT + 12]],
    BLACK_FAINT,
    1
  );
}

function addNodeTaskFC(
  builder: SceneBuilder,
  layerX: number,
  intmData: any,
  cellWidth: number
) {
  const fcLayerFeatures: number[][] | undefined = intmData?.softmax;
  if (!Array.isArray(fcLayerFeatures)) {
    return;
  }

  for (let nodeIndex = 0; nodeIndex < fcLayerFeatures.length; nodeIndex += 1) {
    const feature = fcLayerFeatures[nodeIndex];
    if (Array.isArray(feature)) {
      addSingleFC(builder, layerX, MATRIX_START_Y, feature, nodeIndex, cellWidth);
    }
  }
}

function addEdgeTaskFC(
  builder: SceneBuilder,
  layerX: number,
  queries: number[][],
  cellWidth: number
) {
  const previousLayerX = layerX - LAYER_GAP;
  for (let queryIndex = 0; queryIndex < queries.length; queryIndex += 1) {
    const [nodeA, nodeB] = queries[queryIndex] ?? [];
    if (nodeA == null || nodeB == null) {
      continue;
    }

    const layerYA = MATRIX_START_Y + nodeA * NODE_ROW_HEIGHT + 12;
    const layerYB = MATRIX_START_Y + nodeB * NODE_ROW_HEIGHT + 12;
    const layerYMid = (layerYA + layerYB) / 2;
    const midX = previousLayerX + 50;

    addPolyline(
      builder,
      sampleCubic(
        [previousLayerX, layerYA],
        [midX, layerYA],
        [midX, layerYMid],
        [layerX, layerYMid]
      ),
      BLACK_FAINT,
      1
    );
    addPolyline(
      builder,
      sampleCubic(
        [previousLayerX, layerYB],
        [midX, layerYB],
        [midX, layerYMid],
        [layerX, layerYMid]
      ),
      BLACK_FAINT,
      1
    );

    const probability = Math.random();
    addFeatureVector(
      builder,
      layerX,
      layerYMid - 6,
      [1 - probability, probability],
      cellWidth,
      12,
      1
    );
  }
}

function addGraphTaskFC(
  builder: SceneBuilder,
  layerX: number,
  intmData: any,
  cellWidth: number
) {
  const fcLayerFeatures: number[][] | undefined = intmData?.conv4;
  if (!Array.isArray(fcLayerFeatures) || fcLayerFeatures.length === 0) {
    return;
  }

  const previousLayerX = layerX - LAYER_GAP;
  const midLayerY = MATRIX_START_Y + (fcLayerFeatures.length * NODE_ROW_HEIGHT) / 2;
  for (let nodeIndex = 0; nodeIndex < fcLayerFeatures.length; nodeIndex += 1) {
    const currentY = MATRIX_START_Y + nodeIndex * NODE_ROW_HEIGHT + 12;
    addPolyline(
      builder,
      sampleCubic(
        [previousLayerX, currentY],
        [previousLayerX + 50, currentY],
        [previousLayerX + 50, midLayerY],
        [layerX, midLayerY]
      ),
      BLACK_FAINT,
      1
    );
  }

  let averaged = Array(fcLayerFeatures[0].length).fill(0);
  for (const feature of fcLayerFeatures) {
    averaged = addVector(averaged, feature);
  }
  averaged = divideVector(averaged, fcLayerFeatures.length);

  addFeatureVector(builder, layerX, midLayerY - 6, averaged, cellWidth, 12, 1);
  addSingleFC(builder, layerX + LAYER_GAP, midLayerY - 12, randomVector(4), 0, cellWidth);
}

function addFCByMode(
  builder: SceneBuilder,
  mode: string,
  layerX: number,
  intmData: any,
  queries: number[][],
  cellWidth: number
) {
  if (mode === "node") {
    addNodeTaskFC(builder, layerX, intmData, cellWidth);
    return;
  }

  if (mode === "edge") {
    addEdgeTaskFC(builder, layerX, queries, cellWidth);
    return;
  }

  if (mode === "graph") {
    addGraphTaskFC(builder, layerX, intmData, cellWidth);
  }
}

function translateScene(builder: SceneBuilder): StaticVisualizationScene {
  if (!Number.isFinite(builder.minX) || !Number.isFinite(builder.minY)) {
    return { width: 1, height: 1, primitives: [] };
  }

  const minX = Math.floor(builder.minX - SCENE_PADDING / 2);
  const minY = Math.floor(builder.minY - SCENE_PADDING / 2);
  const width = Math.max(1, Math.ceil(builder.maxX - builder.minX + SCENE_PADDING));
  const height = Math.max(1, Math.ceil(builder.maxY - builder.minY + SCENE_PADDING));

  return {
    width,
    height,
    primitives: builder.primitives.map((primitive) => {
      if (primitive.kind === "rect") {
        return {
          ...primitive,
          x: primitive.x - minX,
          y: primitive.y - minY,
        };
      }

      return {
        ...primitive,
        points: primitive.points.map(([x, y]) => [x - minX, y - minY]),
      };
    }),
  };
}

export function buildVisualizationScene(
  cellWidth: number,
  cellHeight: number,
  adjacencyMatrix: number[][],
  intmData: any,
  links: any[],
  queries: number[][],
  subgraphSample: boolean,
  mode: string
): StaticVisualizationScene {
  const builder = createBuilder();
  const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
  const subgraphData = processSubgraphSequenceDataPipe(adjacencyMatrix, queries, 4);

  addMatrix(builder, adjacencyMatrix);

  if (sortedGNNFeatures.length > 0 && sortedGNNFeatures[0]?.length > 0) {
    const layerX = addIntermediateFeatures(
      builder,
      sortedGNNFeatures,
      cellWidth,
      cellHeight,
      adjacencyMatrix,
      subgraphData,
      subgraphSample
    );
    addBetweenLayerLinks(
      builder,
      sortedGNNFeatures,
      links,
      cellWidth,
      subgraphData,
      subgraphSample
    );
    addFCByMode(builder, mode, layerX, intmData, queries, cellWidth);
  }

  return translateScene(builder);
}

function pushVertex(
  vertices: number[],
  x: number,
  y: number,
  color: Rgba
) {
  vertices.push(x, y, color[0], color[1], color[2], color[3]);
}

function addRectTriangles(
  vertices: number[],
  x: number,
  y: number,
  width: number,
  height: number,
  color: Rgba
) {
  pushVertex(vertices, x, y, color);
  pushVertex(vertices, x + width, y, color);
  pushVertex(vertices, x + width, y + height, color);
  pushVertex(vertices, x, y, color);
  pushVertex(vertices, x + width, y + height, color);
  pushVertex(vertices, x, y + height, color);
}

function addStrokedRectTriangles(
  vertices: number[],
  x: number,
  y: number,
  width: number,
  height: number,
  strokeWidth: number,
  color: Rgba
) {
  const half = strokeWidth / 2;
  addRectTriangles(vertices, x - half, y - half, width + strokeWidth, strokeWidth, color);
  addRectTriangles(vertices, x - half, y + height - half, width + strokeWidth, strokeWidth, color);
  addRectTriangles(vertices, x - half, y - half, strokeWidth, height + strokeWidth, color);
  addRectTriangles(vertices, x + width - half, y - half, strokeWidth, height + strokeWidth, color);
}

function addLineTriangles(
  vertices: number[],
  a: [number, number],
  b: [number, number],
  width: number,
  color: Rgba
) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const length = Math.hypot(dx, dy);
  if (length === 0) {
    return;
  }

  const nx = (-dy / length) * (width / 2);
  const ny = (dx / length) * (width / 2);
  const p1: [number, number] = [a[0] + nx, a[1] + ny];
  const p2: [number, number] = [a[0] - nx, a[1] - ny];
  const p3: [number, number] = [b[0] - nx, b[1] - ny];
  const p4: [number, number] = [b[0] + nx, b[1] + ny];

  pushVertex(vertices, p1[0], p1[1], color);
  pushVertex(vertices, p2[0], p2[1], color);
  pushVertex(vertices, p3[0], p3[1], color);
  pushVertex(vertices, p1[0], p1[1], color);
  pushVertex(vertices, p3[0], p3[1], color);
  pushVertex(vertices, p4[0], p4[1], color);
}

export function buildStaticVisualizationGeometry(scene: StaticVisualizationScene) {
  const vertices: number[] = [];

  for (const primitive of scene.primitives) {
    if (primitive.kind === "rect") {
      if (primitive.fill) {
        addRectTriangles(
          vertices,
          primitive.x,
          primitive.y,
          primitive.width,
          primitive.height,
          primitive.fill
        );
      }

      if (primitive.stroke && primitive.strokeWidth && primitive.strokeWidth > 0) {
        addStrokedRectTriangles(
          vertices,
          primitive.x,
          primitive.y,
          primitive.width,
          primitive.height,
          primitive.strokeWidth,
          primitive.stroke
        );
      }
      continue;
    }

    for (let index = 0; index < primitive.points.length - 1; index += 1) {
      addLineTriangles(
        vertices,
        primitive.points[index],
        primitive.points[index + 1],
        primitive.strokeWidth,
        primitive.stroke
      );
    }
  }

  return new Float32Array(vertices);
}
