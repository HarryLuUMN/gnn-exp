import type {
  StaticBounds,
  StaticExpansionState,
  StaticHoverTarget,
  StaticMatrixLayout,
  StaticVisualizationScene,
} from "./staticScene";
import { curve, featureColor } from "../utils/const";
import { aggregateNeighborFeatures } from "../utils/aggregationUtils";
import {
  extractSortedGNNLayerFeatures,
  getGraphAggregationInfo,
} from "../utils/dataProcessingUtils";
import {
  normalizeLayerBias,
  normalizeLayerWeightMatrix,
} from "../utils/layerRenderingUtils";
import {
  addVector,
  matrixTranspose,
  vecMatMul,
} from "../utils/mathUtils";

type HoverOverlayArgs = {
  canvas: HTMLCanvasElement;
  scene: StaticVisualizationScene;
  adjacencyMatrix: number[][];
  queries: number[][];
  mode: string;
  intmData: any;
  modelInfo: any;
  cellWidth: number;
  expandedFeature?: StaticExpansionState;
  showExpansion?: boolean;
  onExpansionChange?: (expansion: StaticExpansionState) => void;
};

type ScenePoint = {
  x: number;
  y: number;
};

const SVG_NS = "http://www.w3.org/2000/svg";
const HIGHLIGHT = "#203d35";
const MATRIX_HIGHLIGHT = "#000000";
const LINK_HIGHLIGHT = "rgba(32, 61, 53, 0.72)";
const HIT_TOLERANCE = 7;
const FEATURE_HEIGHT = 12;
const DISTANCE_TO_FEATURE = 50;
const WEIGHT_MATRIX_CELL_STROKE = "#d8dedb";
const WEIGHT_MATRIX_CELL_STROKE_WIDTH = 0.35;

function contains(bounds: StaticBounds, point: ScenePoint) {
  return (
    point.x >= bounds.x &&
    point.x <= bounds.x + bounds.width &&
    point.y >= bounds.y &&
    point.y <= bounds.y + bounds.height
  );
}

function distanceToSegment(point: ScenePoint, a: [number, number], b: [number, number]) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq === 0) {
    return Math.hypot(point.x - a[0], point.y - a[1]);
  }

  const t = Math.max(
    0,
    Math.min(1, ((point.x - a[0]) * dx + (point.y - a[1]) * dy) / lengthSq)
  );
  return Math.hypot(point.x - (a[0] + t * dx), point.y - (a[1] + t * dy));
}

function distanceToPolyline(point: ScenePoint, points: Array<[number, number]>) {
  let distance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < points.length - 1; index += 1) {
    distance = Math.min(distance, distanceToSegment(point, points[index], points[index + 1]));
  }
  return distance;
}

function toPath(points: Array<[number, number]>) {
  if (points.length === 0) {
    return "";
  }

  return points
    .map(([x, y], index) => `${index === 0 ? "M" : "L"} ${x} ${y}`)
    .join(" ");
}

function createSvgElement<K extends keyof SVGElementTagNameMap>(
  tagName: K
): SVGElementTagNameMap[K] {
  return document.createElementNS(SVG_NS, tagName);
}

function appendRect(
  group: SVGGElement,
  bounds: StaticBounds,
  options: {
    stroke?: string;
    strokeWidth?: number;
    fill?: string;
    opacity?: number;
  } = {}
) {
  const rect = createSvgElement("rect");
  rect.setAttribute("x", String(bounds.x));
  rect.setAttribute("y", String(bounds.y));
  rect.setAttribute("width", String(bounds.width));
  rect.setAttribute("height", String(bounds.height));
  rect.setAttribute("fill", options.fill ?? "none");
  rect.setAttribute("stroke", options.stroke ?? HIGHLIGHT);
  rect.setAttribute("stroke-width", String(options.strokeWidth ?? 2));
  rect.setAttribute("opacity", String(options.opacity ?? 1));
  group.append(rect);
}

function appendLine(
  group: SVGGElement,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  options: {
    stroke?: string;
    strokeWidth?: number;
  } = {}
) {
  const line = createSvgElement("line");
  line.setAttribute("x1", String(x1));
  line.setAttribute("y1", String(y1));
  line.setAttribute("x2", String(x2));
  line.setAttribute("y2", String(y2));
  line.setAttribute("stroke", options.stroke ?? HIGHLIGHT);
  line.setAttribute("stroke-width", String(options.strokeWidth ?? 1.5));
  line.setAttribute("stroke-linecap", "round");
  group.append(line);
}

function appendPath(
  group: SVGGElement,
  points: Array<[number, number]>,
  options: {
    stroke?: string;
    strokeWidth?: number;
  } = {}
) {
  const path = createSvgElement("path");
  path.setAttribute("d", curve(points) ?? toPath(points));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", options.stroke ?? LINK_HIGHLIGHT);
  path.setAttribute("stroke-width", String(options.strokeWidth ?? 2.5));
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  group.append(path);
}

function appendText(
  group: SVGGElement,
  x: number,
  y: number,
  text: string,
  options: {
    fontSize?: number;
    fill?: string;
    textAnchor?: string;
    transform?: string;
  } = {}
) {
  const node = createSvgElement("text");
  node.setAttribute("x", String(x));
  node.setAttribute("y", String(y));
  node.setAttribute("font-size", String(options.fontSize ?? 6));
  node.setAttribute("fill", options.fill ?? HIGHLIGHT);
  node.setAttribute("text-anchor", options.textAnchor ?? "start");
  if (options.transform) {
    node.setAttribute("transform", options.transform);
  }
  node.textContent = text;
  group.append(node);
}

function appendFeatureLegend(group: SVGGElement, scene: StaticVisualizationScene) {
  const width = 120;
  const height = 7;
  const steps = 40;
  const preferredX = Math.max(
    scene.matrixLayout.x + scene.matrixLayout.width + 60,
    scene.width - width - 40
  );
  const x = Math.max(
    8,
    Math.min(preferredX, Math.max(8, scene.width - width - 8))
  );
  const y = Math.max(12, scene.matrixLayout.y - 30);

  for (let index = 0; index < steps; index += 1) {
    const value = -1.5 + (3 * index) / (steps - 1);
    const rect = createSvgElement("rect");
    rect.setAttribute("x", String(x + (index * width) / steps));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(width / steps + 0.5));
    rect.setAttribute("height", String(height));
    rect.setAttribute("fill", featureColor(value));
    group.append(rect);
  }

  [
    [-1.5, "-1.50"],
    [0, "0.00"],
    [1.5, "1.50"],
  ].forEach(([value, label]) => {
    appendText(
      group,
      x + ((Number(value) + 1.5) / 3) * width,
      y + 20,
      String(label),
      { fontSize: 7, fill: "#6b817b", textAnchor: "middle" }
    );
  });
}

function appendMatrixLabels(group: SVGGElement, layout: StaticMatrixLayout) {
  layout.nodeLabels.forEach((label, index) => {
    const centerY = layout.y + index * layout.cellSize + layout.cellSize / 2 + 3;
    const centerX = layout.x + index * layout.cellSize + layout.cellSize / 2;

    appendText(group, layout.x - 14, centerY, label, {
      fontSize: 8,
      fill: "#203d35",
      textAnchor: "end",
    });
    appendText(group, layout.x - 4, centerY, String(index), {
      fontSize: 6,
      fill: "#6b817b",
      textAnchor: "end",
    });
    appendText(group, centerX, layout.y - 6, label, {
      fontSize: 7,
      fill: "#203d35",
      transform: `rotate(-90 ${centerX} ${layout.y - 6})`,
    });
  });
}

function appendStaticAnnotations(svg: SVGSVGElement, scene: StaticVisualizationScene) {
  const annotations = createSvgElement("g");
  annotations.classList.add("gnn-static-annotations");
  annotations.setAttribute("pointer-events", "none");
  appendMatrixLabels(annotations, scene.matrixLayout);
  appendFeatureLegend(annotations, scene);
  for (const target of scene.hoverTargets) {
    if (target.kind !== "agg-node") {
      continue;
    }

    appendText(
      annotations,
      target.bounds.x,
      target.bounds.y + target.bounds.height + 28,
      target.label,
      { fontSize: 14, fill: "#7f7f7f" }
    );
  }
  svg.append(annotations);
}

function appendIconGroup(
  group: SVGGElement,
  x: number,
  y: number,
  className: string
) {
  const icon = createSvgElement("g");
  icon.setAttribute("class", className);
  icon.setAttribute("transform", `translate(${x - 7}, ${y - 7})`);
  group.append(icon);
  return icon;
}

function appendIconCircle(group: SVGGElement) {
  const circle = createSvgElement("circle");
  circle.setAttribute("cx", "7");
  circle.setAttribute("cy", "7");
  circle.setAttribute("r", "6.5");
  circle.setAttribute("fill", "#ffffff");
  circle.setAttribute("stroke", HIGHLIGHT);
  circle.setAttribute("stroke-width", "0.75");
  group.append(circle);
}

function appendMatmulIcon(group: SVGGElement, x: number, y: number) {
  const icon = appendIconGroup(group, x, y, "matmul-icon layer-inner-works");
  appendIconCircle(icon);

  for (let row = 0; row < 2; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      appendRect(
        icon,
        {
          x: 2.3 + row * 1.6,
          y: 3.2 + col * 1.6,
          width: 1.2,
          height: 1.2,
        },
        { stroke: HIGHLIGHT, strokeWidth: 0.35, fill: "#ffffff" }
      );
    }
  }

  const text = createSvgElement("text");
  text.setAttribute("x", "9.4");
  text.setAttribute("y", "8.7");
  text.setAttribute("text-anchor", "middle");
  text.setAttribute("font-size", "4.2");
  text.setAttribute("fill", HIGHLIGHT);
  text.textContent = "x";
  icon.append(text);
}

function appendActivationIcon(group: SVGGElement, x: number, y: number) {
  const icon = appendIconGroup(group, x, y, "activation-icon layer-inner-works");
  appendIconCircle(icon);

  const axis = createSvgElement("path");
  axis.setAttribute("d", "M2.2 9.3 H11.8 M7 2.2 V11.8");
  axis.setAttribute("stroke", "rgba(32, 61, 53, 0.34)");
  axis.setAttribute("stroke-width", "0.35");
  axis.setAttribute("fill", "none");
  icon.append(axis);

  const relu = createSvgElement("path");
  relu.setAttribute("d", "M2.4 9.2 H6.4 L11.5 4.1");
  relu.setAttribute("stroke", "#c6501d");
  relu.setAttribute("stroke-width", "0.9");
  relu.setAttribute("stroke-linecap", "round");
  relu.setAttribute("stroke-linejoin", "round");
  relu.setAttribute("fill", "none");
  icon.append(relu);
}

function appendFeatureVector(
  group: SVGGElement,
  x: number,
  y: number,
  vector: number[],
  cellWidth: number,
  cellHeight: number = FEATURE_HEIGHT
) {
  appendRect(group, {
    x,
    y,
    width: vector.length * cellWidth,
    height: cellHeight,
  });

  for (let index = 0; index < vector.length; index += 1) {
    appendRect(
      group,
      {
        x: x + index * cellWidth,
        y,
        width: cellWidth,
        height: cellHeight,
      },
      {
        fill: featureColor(vector[index]),
        stroke: "rgba(80, 80, 80, 0.45)",
        strokeWidth: 0.5,
      }
    );
  }
}

function matrixCellAt(layout: StaticMatrixLayout, point: ScenePoint) {
  if (!contains(layout, point) || layout.cellSize <= 0) {
    return null;
  }

  const col = Math.floor((point.x - layout.x) / layout.cellSize);
  const row = Math.floor((point.y - layout.y) / layout.cellSize);
  if (row < 0 || col < 0 || row >= layout.nodeCount || col >= layout.nodeCount) {
    return null;
  }

  return { row, col };
}

function matrixRowBounds(layout: StaticMatrixLayout, row: number): StaticBounds {
  return {
    x: layout.x,
    y: layout.y + row * layout.cellSize,
    width: layout.width,
    height: layout.cellSize,
  };
}

function matrixColBounds(layout: StaticMatrixLayout, col: number): StaticBounds {
  return {
    x: layout.x + col * layout.cellSize,
    y: layout.y,
    width: layout.cellSize,
    height: layout.height,
  };
}

function frameTarget(
  targets: StaticHoverTarget[],
  layerIndex: number,
  nodeIndex: number
) {
  return targets.find(
    (target) =>
      target.kind === "feature-node" &&
      target.layerIndex === layerIndex &&
      target.nodeIndex === nodeIndex
  ) as Extract<StaticHoverTarget, { kind: "feature-node" }> | undefined;
}

function highlightFeatureFrame(
  group: SVGGElement,
  scene: StaticVisualizationScene,
  layerIndex: number,
  nodeIndex: number,
  opacity: number = 0.8
) {
  const target = frameTarget(scene.hoverTargets, layerIndex, nodeIndex);
  if (target && "bounds" in target) {
    appendRect(group, target.bounds, { opacity });
  }
}

function getScenePoint(svg: SVGSVGElement, scene: StaticVisualizationScene, event: PointerEvent) {
  const rect = svg.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / Math.max(rect.width, 1)) * scene.width,
    y: ((event.clientY - rect.top) / Math.max(rect.height, 1)) * scene.height,
  };
}

function hitTest(scene: StaticVisualizationScene, point: ScenePoint) {
  for (let index = scene.hoverTargets.length - 1; index >= 0; index -= 1) {
    const target = scene.hoverTargets[index];
    if ("bounds" in target && contains(target.bounds, point)) {
      return target;
    }
  }

  for (let index = scene.hoverTargets.length - 1; index >= 0; index -= 1) {
    const target = scene.hoverTargets[index];
    if ("points" in target && distanceToPolyline(point, target.points) <= HIT_TOLERANCE) {
      return target;
    }
  }

  return null;
}

function drawMatrixActivation(
  group: SVGGElement,
  matrixLayout: StaticMatrixLayout,
  row: number,
  col?: number
) {
  appendRect(group, matrixRowBounds(matrixLayout, row), {
    stroke: MATRIX_HIGHLIGHT,
    strokeWidth: 2,
  });

  if (col != null) {
    appendRect(group, matrixColBounds(matrixLayout, col), {
      stroke: MATRIX_HIGHLIGHT,
      strokeWidth: 2,
    });
  }
}

function drawFeatureHover(
  group: SVGGElement,
  target: Extract<StaticHoverTarget, { kind: "feature-node" }>,
  scene: StaticVisualizationScene,
  adjacencyMatrix: number[][]
) {
  appendRect(group, target.bounds);
  if (target.layerIndex === 0) {
    drawMatrixActivation(group, scene.matrixLayout, target.nodeIndex);
    return;
  }

  drawMatrixActivation(group, scene.matrixLayout, target.nodeIndex, target.nodeIndex);
  const links = scene.hoverTargets.filter(
    (
      candidate
    ): candidate is Extract<StaticHoverTarget, { kind: "layer-link" }> =>
      candidate.kind === "layer-link" &&
      candidate.layerIndex === target.layerIndex - 1 &&
      candidate.targetIndex === target.nodeIndex
  );

  for (const link of links) {
    appendPath(group, link.points);
    const source = frameTarget(scene.hoverTargets, target.layerIndex - 1, link.sourceIndex);
    if (source && "bounds" in source) {
      appendRect(group, source.bounds, { opacity: 0.8 });
    }
  }

  const selfLink = scene.hoverTargets.find(
    (candidate) =>
      candidate.kind === "self-link" &&
      candidate.layerIndex === target.layerIndex - 1 &&
      candidate.nodeIndex === target.nodeIndex
  );
  if (selfLink && "points" in selfLink) {
    appendPath(group, selfLink.points);
    highlightFeatureFrame(
      group,
      scene,
      target.layerIndex - 1,
      target.nodeIndex,
      0.8
    );
  }

  const adjacencyRow = adjacencyMatrix[target.nodeIndex] ?? [];
  for (let index = 0; index < adjacencyRow.length; index += 1) {
    if (adjacencyRow[index] === 1 && index !== target.nodeIndex) {
      appendRect(group, matrixRowBounds(scene.matrixLayout, index), {
        stroke: MATRIX_HIGHLIGHT,
        strokeWidth: 1.5,
        opacity: 0.65,
      });
    }
  }
}

function drawFcHover(
  group: SVGGElement,
  target: Extract<StaticHoverTarget, { kind: "fc-node" }>,
  scene: StaticVisualizationScene,
  queries: number[][],
  mode: string
) {
  appendRect(group, target.bounds);
  for (const candidate of scene.hoverTargets) {
    if (candidate.kind === "fc-link" && candidate.nodeIndex === target.nodeIndex) {
      appendPath(group, candidate.points);
    }
  }

  if (mode === "edge") {
    const query = queries[target.nodeIndex] ?? [];
    for (const node of query) {
      drawMatrixActivation(group, scene.matrixLayout, node);
    }
    return;
  }

  drawMatrixActivation(group, scene.matrixLayout, target.nodeIndex);
}

function drawAggHover(
  group: SVGGElement,
  target: Extract<StaticHoverTarget, { kind: "agg-node" }>,
  scene: StaticVisualizationScene
) {
  appendRect(group, target.bounds);
  for (const candidate of scene.hoverTargets) {
    if (candidate.kind === "agg-link") {
      appendPath(group, candidate.points);
    }
  }
}

function drawLinkHover(
  group: SVGGElement,
  target: Extract<StaticHoverTarget, { points: Array<[number, number]> }>
) {
  appendPath(group, target.points, { strokeWidth: 3 });
}

function drawWeightMatrix(
  group: SVGGElement,
  x: number,
  y: number,
  matrix: number[][],
  cellWidth: number,
  dirCoefficient: number
) {
  if (matrix.length === 0 || matrix[0].length === 0) {
    return;
  }

  const frameY = y + (dirCoefficient < 0 ? -cellWidth * (matrix.length - 1) : 0);
  appendRect(group, {
    x,
    y: frameY,
    width: matrix[0].length * cellWidth,
    height: matrix.length * cellWidth,
  });

  for (let row = 0; row < matrix.length; row += 1) {
    for (let col = 0; col < matrix[row].length; col += 1) {
      appendRect(
        group,
        {
          x: x + col * cellWidth,
          y: y + dirCoefficient * row * cellWidth,
          width: cellWidth,
          height: cellWidth,
        },
        {
          fill: featureColor(matrix[row][col]),
          stroke: WEIGHT_MATRIX_CELL_STROKE,
          strokeWidth: WEIGHT_MATRIX_CELL_STROKE_WIDTH,
        }
      );
    }
  }
}

function drawTopDownWeightMatrix(
  group: SVGGElement,
  x: number,
  y: number,
  matrix: number[][],
  cellWidth: number
) {
  if (matrix.length === 0 || matrix[0].length === 0) {
    return;
  }

  appendRect(group, {
    x,
    y,
    width: matrix[0].length * cellWidth,
    height: matrix.length * cellWidth,
  });

  for (let row = 0; row < matrix.length; row += 1) {
    for (let col = 0; col < matrix[row].length; col += 1) {
      appendRect(
        group,
        {
          x: x + col * cellWidth,
          y: y + row * cellWidth,
          width: cellWidth,
          height: cellWidth,
        },
        {
          fill: featureColor(matrix[row][col]),
          stroke: WEIGHT_MATRIX_CELL_STROKE,
          strokeWidth: WEIGHT_MATRIX_CELL_STROKE_WIDTH,
        }
      );
    }
  }
}

function layerRightBoundary(targets: StaticHoverTarget[], layerIndex: number) {
  let right = Number.NEGATIVE_INFINITY;
  for (const target of targets) {
    if (target.kind === "feature-node" && target.layerIndex === layerIndex) {
      right = Math.max(right, target.bounds.x + target.bounds.width);
    }
  }

  return Number.isFinite(right) ? right : null;
}

function fcTarget(
  targets: StaticHoverTarget[],
  nodeIndex: number
) {
  return targets.find(
    (target) => target.kind === "fc-node" && target.nodeIndex === nodeIndex
  ) as Extract<StaticHoverTarget, { kind: "fc-node" }> | undefined;
}

function drawFeatureExpansion(
  group: SVGGElement,
  target: Extract<StaticHoverTarget, { kind: "feature-node" }>,
  scene: StaticVisualizationScene,
  adjacencyMatrix: number[][],
  sortedGNNFeatures: number[][][],
  modelInfo: any,
  cellWidth: number,
  expansion: StaticExpansionState
) {
  if (target.layerIndex === 0) {
    return;
  }

  const layerInfo = modelInfo?.[`conv${target.layerIndex}`];
  if (!layerInfo) {
    return;
  }

  const dirCoefficient = target.nodeIndex < scene.matrixLayout.nodeCount / 2 ? 1 : -1;
  const currentNodeX =
    layerRightBoundary(scene.hoverTargets, target.layerIndex - 1) ??
    target.bounds.x -
      (expansion?.kind === "gnn" ? expansion.distance : 0) -
      DISTANCE_TO_FEATURE;
  const currentNodeY = target.bounds.y + target.bounds.height / 2;
  const aggregationResult = aggregateNeighborFeatures(
    adjacencyMatrix,
    sortedGNNFeatures,
    target.layerIndex,
    target.nodeIndex,
    layerInfo
  );
  const aggregatedFeature = aggregationResult.aggregatedFeature;
  const firstIntersect: [number, number] = [
    currentNodeX + DISTANCE_TO_FEATURE,
    currentNodeY,
  ];

  for (const { nodeIndex, label } of aggregationResult.contributions) {
    const previous = frameTarget(scene.hoverTargets, target.layerIndex - 1, nodeIndex);
    if (!previous || !("bounds" in previous)) {
      continue;
    }

    const sourceY = previous.bounds.y + previous.bounds.height / 2;
    const sourceX = previous.bounds.x + previous.bounds.width;
    const controlX = sourceX + (firstIntersect[0] - sourceX) / 2;
    const ctrlPointForTargetNode: [number, number] = [
      controlX,
      sourceY,
    ];
    const ctrlPointForCurrentNode: [number, number] = [
      controlX,
      currentNodeY,
    ];
    appendPath(
      group,
      [
        [sourceX, sourceY],
        ctrlPointForTargetNode,
        ctrlPointForCurrentNode,
        firstIntersect,
      ],
      { stroke: HIGHLIGHT, strokeWidth: 1.5 }
    );
    appendText(group, sourceX + 3, sourceY - 6, label);
    appendRect(group, previous.bounds, { opacity: 0.85 });
  }

  appendRect(group, target.bounds, { strokeWidth: 2.5 });
  const aggregatedX = currentNodeX + DISTANCE_TO_FEATURE;
  const aggregatedY = currentNodeY - FEATURE_HEIGHT / 2;
  appendText(group, aggregatedX, aggregatedY - 4, `agg: ${aggregationResult.label}`, {
    fontSize: 7,
  });
  appendFeatureVector(group, aggregatedX, aggregatedY, aggregatedFeature, cellWidth);

  const weightMatrix = normalizeLayerWeightMatrix(layerInfo, aggregatedFeature.length);
  if (!weightMatrix) {
    return;
  }
  const aggregatedRight = aggregatedX + aggregatedFeature.length * cellWidth;
  const multipliedX = aggregatedRight + DISTANCE_TO_FEATURE;
  const matrixX =
    aggregatedRight -
    ((weightMatrix[0]?.length ?? 0) * cellWidth) / 2;
  const matrixY = currentNodeY + dirCoefficient * DISTANCE_TO_FEATURE;

  appendLine(
    group,
    aggregatedRight,
    currentNodeY,
    multipliedX,
    currentNodeY
  );
  appendMatmulIcon(
    group,
    aggregatedRight + DISTANCE_TO_FEATURE / 2,
    currentNodeY
  );
  appendPath(
    group,
    [
      [aggregatedRight + DISTANCE_TO_FEATURE / 2, currentNodeY],
      [
        aggregatedRight + DISTANCE_TO_FEATURE / 2,
        currentNodeY + dirCoefficient * DISTANCE_TO_FEATURE * 0.5,
      ],
      [
        aggregatedRight,
        currentNodeY + dirCoefficient * DISTANCE_TO_FEATURE * 0.5,
      ],
      [
        aggregatedRight,
        currentNodeY + dirCoefficient * DISTANCE_TO_FEATURE,
      ],
    ],
    { stroke: HIGHLIGHT, strokeWidth: 1.4 }
  );
  drawWeightMatrix(group, matrixX, matrixY, weightMatrix, cellWidth, dirCoefficient);

  let multipliedFeature: number[] = [];
  try {
    multipliedFeature = vecMatMul(aggregatedFeature, weightMatrix);
  } catch {
    multipliedFeature = Array(weightMatrix[0]?.length ?? 0).fill(0);
  }
  const bias = normalizeLayerBias(layerInfo, multipliedFeature.length);

  appendFeatureVector(
    group,
    multipliedX,
    currentNodeY - FEATURE_HEIGHT / 2,
    multipliedFeature,
    cellWidth
  );
  appendFeatureVector(
    group,
    multipliedX,
    currentNodeY - dirCoefficient * DISTANCE_TO_FEATURE,
    bias,
    cellWidth
  );
  appendLine(
    group,
    multipliedX + multipliedFeature.length * cellWidth,
    currentNodeY,
    target.bounds.x,
    currentNodeY
  );
  appendActivationIcon(
    group,
    multipliedX + multipliedFeature.length * cellWidth + DISTANCE_TO_FEATURE / 2,
    currentNodeY
  );
  appendPath(
    group,
    [
      [
        multipliedX + bias.length * cellWidth,
        currentNodeY - dirCoefficient * DISTANCE_TO_FEATURE + FEATURE_HEIGHT / 2,
      ],
      [
        multipliedX + bias.length * cellWidth + DISTANCE_TO_FEATURE / 2,
        currentNodeY - dirCoefficient * DISTANCE_TO_FEATURE + FEATURE_HEIGHT / 2,
      ],
      [
        multipliedX + bias.length * cellWidth + DISTANCE_TO_FEATURE / 2,
        currentNodeY,
      ],
      [multipliedX + bias.length * cellWidth + DISTANCE_TO_FEATURE, currentNodeY],
    ],
    { stroke: HIGHLIGHT, strokeWidth: 1.4 }
  );
}

function drawFcExpansion(
  group: SVGGElement,
  target: Extract<StaticHoverTarget, { kind: "fc-node" }>,
  sortedGNNFeatures: number[][][],
  modelInfo: any,
  intmData: any,
  cellWidth: number,
  mode: string,
  expansion: StaticExpansionState
) {
  const classifier = modelInfo?.classifier;
  if (!classifier?.weight) {
    return;
  }

  const lastLayer = sortedGNNFeatures[sortedGNNFeatures.length - 1] ?? [];
  const graphAggregation = getGraphAggregationInfo(intmData, lastLayer);
  const inputFeature =
    mode === "graph"
      ? graphAggregation?.feature ?? []
      : lastLayer[target.nodeIndex] ?? [];
  if (inputFeature.length === 0) {
    return;
  }

  const originalNodeX =
    target.bounds.x - (expansion?.kind === "fc" ? expansion.distance : 0);
  const currentNodeY = target.bounds.y + target.bounds.height / 2;
  const dirCoefficient =
    target.nodeIndex < Math.max(1, lastLayer.length) / 2 ? 1 : -1;
  const weightMatrix = matrixTranspose(classifier.weight as number[][]);
  let multipliedFeature: number[] = [];
  try {
    multipliedFeature = vecMatMul(inputFeature, weightMatrix);
  } catch {
    multipliedFeature = Array(weightMatrix[0]?.length ?? 0).fill(0);
  }
  const bias = Array.isArray(classifier.bias)
    ? (classifier.bias as number[])
    : Array(multipliedFeature.length).fill(0);
  const biasedAddition =
    bias.length === multipliedFeature.length
      ? addVector(multipliedFeature, bias)
      : multipliedFeature;

  const matrixWidth = (weightMatrix[0]?.length ?? 0) * cellWidth;
  const additionWidth = biasedAddition.length * cellWidth;
  const additionX = target.bounds.x - DISTANCE_TO_FEATURE - additionWidth;
  const matrixStartX = additionX - DISTANCE_TO_FEATURE - multipliedFeature.length * cellWidth;
  const currentNodeX = Math.min(originalNodeX, matrixStartX - DISTANCE_TO_FEATURE);
  const matrixStartY = currentNodeY - dirCoefficient * DISTANCE_TO_FEATURE;

  appendRect(group, target.bounds, { strokeWidth: 2.5 });
  appendLine(
    group,
    currentNodeX,
    currentNodeY,
    currentNodeX + DISTANCE_TO_FEATURE,
    currentNodeY
  );
  appendMatmulIcon(
    group,
    currentNodeX + DISTANCE_TO_FEATURE / 2,
    currentNodeY
  );
  appendPath(
    group,
    [
      [currentNodeX + DISTANCE_TO_FEATURE / 2, currentNodeY],
      [
        currentNodeX + DISTANCE_TO_FEATURE / 2,
        currentNodeY - (DISTANCE_TO_FEATURE / 2) * dirCoefficient,
      ],
      [
        matrixStartX + matrixWidth / 2,
        currentNodeY - (DISTANCE_TO_FEATURE / 2) * dirCoefficient,
      ],
      [
        matrixStartX + matrixWidth / 2,
        matrixStartY,
      ],
    ],
    { stroke: HIGHLIGHT, strokeWidth: 1.4 }
  );
  drawTopDownWeightMatrix(group, matrixStartX, matrixStartY, weightMatrix, cellWidth);

  appendFeatureVector(
    group,
    matrixStartX,
    currentNodeY - FEATURE_HEIGHT / 2,
    multipliedFeature,
    cellWidth
  );
  appendLine(
    group,
    matrixStartX + multipliedFeature.length * cellWidth,
    currentNodeY,
    matrixStartX + DISTANCE_TO_FEATURE + multipliedFeature.length * cellWidth,
    currentNodeY
  );
  appendFeatureVector(
    group,
    matrixStartX,
    currentNodeY + (dirCoefficient * DISTANCE_TO_FEATURE) / 2 - FEATURE_HEIGHT / 2,
    bias,
    cellWidth
  );
  appendPath(
    group,
    [
      [
        matrixStartX + multipliedFeature.length * cellWidth,
        currentNodeY + (dirCoefficient * DISTANCE_TO_FEATURE) / 2,
      ],
      [
        matrixStartX + DISTANCE_TO_FEATURE / 2 + multipliedFeature.length * cellWidth,
        currentNodeY + (dirCoefficient * DISTANCE_TO_FEATURE) / 2,
      ],
      [
        matrixStartX + DISTANCE_TO_FEATURE / 2 + multipliedFeature.length * cellWidth,
        currentNodeY,
      ],
      [
        matrixStartX + DISTANCE_TO_FEATURE + multipliedFeature.length * cellWidth,
        currentNodeY,
      ],
    ],
    { stroke: HIGHLIGHT, strokeWidth: 1.4 }
  );
  appendFeatureVector(
    group,
    additionX,
    currentNodeY - FEATURE_HEIGHT / 2,
    biasedAddition,
    cellWidth
  );
  appendLine(
    group,
    additionX + biasedAddition.length * cellWidth,
    currentNodeY,
    target.bounds.x,
    currentNodeY
  );
  appendActivationIcon(
    group,
    additionX + biasedAddition.length * cellWidth + DISTANCE_TO_FEATURE / 2,
    currentNodeY
  );
}

function drawExpansion(
  group: SVGGElement,
  target:
    | Extract<StaticHoverTarget, { kind: "feature-node" }>
    | Extract<StaticHoverTarget, { kind: "fc-node" }>
    | null,
  scene: StaticVisualizationScene,
  adjacencyMatrix: number[][],
  sortedGNNFeatures: number[][][],
  modelInfo: any,
  intmData: any,
  cellWidth: number,
  mode: string,
  expansion: StaticExpansionState
) {
  group.replaceChildren();
  if (!target) {
    return;
  }

  if (target.kind === "fc-node") {
    drawFcExpansion(
      group,
      target,
      sortedGNNFeatures,
      modelInfo,
      intmData,
      cellWidth,
      mode,
      expansion
    );
    return;
  }

  drawFeatureExpansion(
    group,
    target,
    scene,
    adjacencyMatrix,
    sortedGNNFeatures,
    modelInfo,
    cellWidth,
    expansion
  );
}

function getExpansionDistance(
  sortedGNNFeatures: number[][][],
  layerIndex: number,
  cellWidth: number
) {
  const previousWidth = (sortedGNNFeatures[layerIndex - 1]?.[0]?.length ?? 0) * cellWidth;
  const currentWidth = (sortedGNNFeatures[layerIndex]?.[0]?.length ?? 0) * cellWidth;
  return DISTANCE_TO_FEATURE * 3 + previousWidth + currentWidth - 100;
}

function getFcExpansionDistance(
  sortedGNNFeatures: number[][][],
  modelInfo: any,
  cellWidth: number
) {
  const lastWidth =
    (sortedGNNFeatures[sortedGNNFeatures.length - 1]?.[0]?.length ?? 0) *
    cellWidth;
  const biasWidth = (modelInfo?.classifier?.bias?.length ?? 4) * cellWidth;
  return DISTANCE_TO_FEATURE * 3 + lastWidth + biasWidth * 2 - 100;
}

function drawHover(
  group: SVGGElement,
  target: StaticHoverTarget | null,
  scene: StaticVisualizationScene,
  adjacencyMatrix: number[][],
  queries: number[][],
  mode: string
) {
  group.replaceChildren();

  if (!target) {
    return;
  }

  if (target.kind === "feature-node") {
    drawFeatureHover(group, target, scene, adjacencyMatrix);
    return;
  }

  if (target.kind === "fc-node") {
    drawFcHover(group, target, scene, queries, mode);
    return;
  }

  if (target.kind === "agg-node") {
    drawAggHover(group, target, scene);
    return;
  }

  drawLinkHover(group, target);
}

export function attachStaticHoverOverlay({
  canvas,
  scene,
  adjacencyMatrix,
  queries,
  mode,
  intmData,
  modelInfo,
  cellWidth,
  expandedFeature,
  showExpansion = true,
  onExpansionChange,
}: HoverOverlayArgs) {
  const wrapper = document.createElement("div");
  wrapper.className = "gnn-static-gpu-layer";
  wrapper.style.width = `${Math.max(1, Math.ceil(scene.width))}px`;
  wrapper.style.height = `${Math.max(1, Math.ceil(scene.height))}px`;

  const parent = canvas.parentElement;
  parent?.insertBefore(wrapper, canvas);
  wrapper.append(canvas);

  const svg = createSvgElement("svg");
  svg.classList.add("gnn-static-hover-overlay");
  svg.setAttribute("width", String(Math.max(1, Math.ceil(scene.width))));
  svg.setAttribute("height", String(Math.max(1, Math.ceil(scene.height))));
  svg.setAttribute("viewBox", `0 0 ${scene.width} ${scene.height}`);
  appendStaticAnnotations(svg, scene);

  const group = createSvgElement("g");
  group.classList.add("gnn-static-hover-marks");
  svg.append(group);
  const expansionGroup = createSvgElement("g");
  expansionGroup.classList.add("gnn-static-expansion-marks");
  svg.append(expansionGroup);
  wrapper.append(svg);
  const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
  let expandedTarget:
    | Extract<StaticHoverTarget, { kind: "feature-node" }>
    | Extract<StaticHoverTarget, { kind: "fc-node" }>
    | null =
    expandedFeature?.kind === "gnn" && expandedFeature.layerIndex > 0
      ? frameTarget(scene.hoverTargets, expandedFeature.layerIndex, expandedFeature.nodeIndex) ?? null
      : expandedFeature?.kind === "fc"
        ? fcTarget(scene.hoverTargets, expandedFeature.nodeIndex) ?? null
        : null;

  if (expandedTarget && showExpansion) {
    drawExpansion(
      expansionGroup,
      expandedTarget,
      scene,
      adjacencyMatrix,
      sortedGNNFeatures,
      modelInfo,
      intmData,
      cellWidth,
      mode,
      expandedFeature ?? null
    );
  }

  const clearExpansion = () => {
    if (onExpansionChange) {
      onExpansionChange(null);
      return;
    }

    expandedTarget = null;
    expansionGroup.replaceChildren();
  };

  const onPointerMove = (event: PointerEvent) => {
    if (expandedTarget) {
      group.replaceChildren();
      return;
    }

    const point = getScenePoint(svg, scene, event);
    const matrixCell = matrixCellAt(scene.matrixLayout, point);
    if (matrixCell) {
      group.replaceChildren();
      drawMatrixActivation(group, scene.matrixLayout, matrixCell.row, matrixCell.col);
      return;
    }

    drawHover(group, hitTest(scene, point), scene, adjacencyMatrix, queries, mode);
  };

  const onPointerDown = (event: PointerEvent) => {
    if (expandedTarget) {
      clearExpansion();
      return;
    }

    const point = getScenePoint(svg, scene, event);
    const target = hitTest(scene, point);
    if (target?.kind === "feature-node" && target.layerIndex > 0) {
      const nextExpansion = {
        kind: "gnn" as const,
        layerIndex: target.layerIndex,
        nodeIndex: target.nodeIndex,
        distance: getExpansionDistance(sortedGNNFeatures, target.layerIndex, cellWidth),
      };

      if (onExpansionChange) {
        onExpansionChange(nextExpansion);
        return;
      }

      expandedTarget = target;
      drawExpansion(
        expansionGroup,
        expandedTarget,
        scene,
        adjacencyMatrix,
        sortedGNNFeatures,
        modelInfo,
        intmData,
        cellWidth,
        mode,
        nextExpansion
      );
      return;
    }

    if (target?.kind === "fc-node") {
      const nextExpansion = {
        kind: "fc" as const,
        nodeIndex: target.nodeIndex,
        distance: getFcExpansionDistance(sortedGNNFeatures, modelInfo, cellWidth),
      };

      if (onExpansionChange) {
        onExpansionChange(nextExpansion);
        return;
      }

      expandedTarget = target;
      drawExpansion(
        expansionGroup,
        expandedTarget,
        scene,
        adjacencyMatrix,
        sortedGNNFeatures,
        modelInfo,
        intmData,
        cellWidth,
        mode,
        nextExpansion
      );
      return;
    }

    if (!target) {
      clearExpansion();
    }
  };

  const onPointerLeave = () => {
    group.replaceChildren();
  };

  svg.addEventListener("pointermove", onPointerMove);
  svg.addEventListener("pointerdown", onPointerDown);
  svg.addEventListener("pointerleave", onPointerLeave);

  return {
    overlay: wrapper,
    destroy() {
      svg.removeEventListener("pointermove", onPointerMove);
      svg.removeEventListener("pointerdown", onPointerDown);
      svg.removeEventListener("pointerleave", onPointerLeave);
      group.replaceChildren();
      expansionGroup.replaceChildren();
      wrapper.remove();
    },
  };
}
