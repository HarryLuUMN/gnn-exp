import type {
  StaticBounds,
  StaticHoverTarget,
  StaticMatrixLayout,
  StaticVisualizationScene,
} from "./staticScene";
import { featureColor } from "../utils/const";
import { extractSortedGNNLayerFeatures } from "../utils/dataProcessingUtils";
import {
  addVector,
  countOnes,
  matrixTranspose,
  scaleVector,
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
  path.setAttribute("d", toPath(points));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", options.stroke ?? LINK_HIGHLIGHT);
  path.setAttribute("stroke-width", String(options.strokeWidth ?? 2.5));
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  group.append(path);
}

function appendText(group: SVGGElement, x: number, y: number, text: string) {
  const node = createSvgElement("text");
  node.setAttribute("x", String(x));
  node.setAttribute("y", String(y));
  node.setAttribute("font-size", "6");
  node.setAttribute("fill", HIGHLIGHT);
  node.textContent = text;
  group.append(node);
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
  );
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
          stroke: "rgba(80, 80, 80, 0.35)",
          strokeWidth: 0.4,
        }
      );
    }
  }
}

function safeAddVector(a: number[], b: number[]) {
  if (a.length !== b.length) {
    return a;
  }

  return addVector(a, b);
}

function buildAggregatedFeature(
  adjacencyMatrix: number[][],
  sortedGNNFeatures: number[][][],
  layerIndex: number,
  nodeIndex: number
) {
  const previousLayer = sortedGNNFeatures[layerIndex - 1] ?? [];
  const featureLength = previousLayer[0]?.length ?? 0;
  let aggregatedFeature = Array(featureLength).fill(0);
  const degreeMultipliers: Array<{ nodeIndex: number; value: number }> = [];

  const adjacencyRow = adjacencyMatrix[nodeIndex] ?? [];
  for (let index = 0; index < adjacencyRow.length; index += 1) {
    if (adjacencyRow[index] !== 1 || !previousLayer[index]) {
      continue;
    }

    const degreeMultiplier =
      1 /
      Math.sqrt(
        Math.max(1, countOnes(adjacencyMatrix[nodeIndex])) *
          Math.max(1, countOnes(adjacencyMatrix[index] ?? []))
      );
    aggregatedFeature = safeAddVector(
      aggregatedFeature,
      scaleVector(degreeMultiplier, previousLayer[index])
    );
    degreeMultipliers.push({ nodeIndex: index, value: degreeMultiplier });
  }

  return { aggregatedFeature, degreeMultipliers };
}

function drawFeatureExpansion(
  group: SVGGElement,
  target: Extract<StaticHoverTarget, { kind: "feature-node" }>,
  scene: StaticVisualizationScene,
  adjacencyMatrix: number[][],
  sortedGNNFeatures: number[][][],
  modelInfo: any,
  cellWidth: number
) {
  if (target.layerIndex === 0) {
    return;
  }

  const layerInfo = modelInfo?.[`conv${target.layerIndex}`];
  if (!layerInfo?.weight) {
    return;
  }

  const direction = target.nodeIndex < scene.matrixLayout.nodeCount / 2 ? 1 : -1;
  const currentNodeX = target.bounds.x;
  const currentNodeY = target.bounds.y + target.bounds.height / 2;
  const { aggregatedFeature, degreeMultipliers } = buildAggregatedFeature(
    adjacencyMatrix,
    sortedGNNFeatures,
    target.layerIndex,
    target.nodeIndex
  );
  const firstIntersect: [number, number] = [
    currentNodeX + DISTANCE_TO_FEATURE,
    currentNodeY,
  ];

  for (const { nodeIndex, value } of degreeMultipliers) {
    const previous = frameTarget(scene.hoverTargets, target.layerIndex - 1, nodeIndex);
    if (!previous || !("bounds" in previous)) {
      continue;
    }

    const sourceY = previous.bounds.y + previous.bounds.height / 2;
    appendPath(
      group,
      [
        [previous.bounds.x + previous.bounds.width, sourceY],
        [currentNodeX + DISTANCE_TO_FEATURE / 2, sourceY],
        [currentNodeX + DISTANCE_TO_FEATURE / 2, currentNodeY],
        firstIntersect,
      ],
      { stroke: HIGHLIGHT, strokeWidth: 1.5 }
    );
    appendText(group, previous.bounds.x + previous.bounds.width + 3, sourceY - 5, value.toFixed(2));
    appendRect(group, previous.bounds, { opacity: 0.85 });
  }

  appendRect(group, target.bounds, { strokeWidth: 2.5 });
  const aggregatedX = currentNodeX + DISTANCE_TO_FEATURE;
  const aggregatedY = currentNodeY - FEATURE_HEIGHT / 2;
  appendFeatureVector(group, aggregatedX, aggregatedY, aggregatedFeature, cellWidth);

  const weightMatrix = matrixTranspose(layerInfo.weight as number[][]);
  const matrixX =
    currentNodeX +
    DISTANCE_TO_FEATURE * 1.5 +
    aggregatedFeature.length * cellWidth -
    DISTANCE_TO_FEATURE * 0.5 -
    ((weightMatrix[0]?.length ?? 0) * cellWidth) / 2;
  const matrixY =
    currentNodeY +
    direction * DISTANCE_TO_FEATURE -
    (direction < 0 ? Math.max(0, weightMatrix.length - 1) * cellWidth : 0);

  appendLine(
    group,
    aggregatedX + aggregatedFeature.length * cellWidth,
    currentNodeY,
    aggregatedX + aggregatedFeature.length * cellWidth + DISTANCE_TO_FEATURE,
    currentNodeY
  );
  appendPath(
    group,
    [
      [currentNodeX + DISTANCE_TO_FEATURE * 1.5 + aggregatedFeature.length * cellWidth, currentNodeY],
      [
        currentNodeX + DISTANCE_TO_FEATURE * 1.5 + aggregatedFeature.length * cellWidth,
        currentNodeY + direction * DISTANCE_TO_FEATURE * 0.5,
      ],
      [
        currentNodeX + DISTANCE_TO_FEATURE + aggregatedFeature.length * cellWidth,
        currentNodeY + direction * DISTANCE_TO_FEATURE * 0.5,
      ],
      [
        currentNodeX + DISTANCE_TO_FEATURE + aggregatedFeature.length * cellWidth,
        currentNodeY + direction * DISTANCE_TO_FEATURE,
      ],
    ],
    { stroke: HIGHLIGHT, strokeWidth: 1.4 }
  );
  drawWeightMatrix(group, matrixX, matrixY, weightMatrix, cellWidth);

  let multipliedFeature: number[] = [];
  try {
    multipliedFeature = vecMatMul(aggregatedFeature, weightMatrix);
  } catch {
    multipliedFeature = Array(weightMatrix[0]?.length ?? 0).fill(0);
  }
  const bias = Array.isArray(layerInfo.bias)
    ? (layerInfo.bias as number[])
    : Array(multipliedFeature.length).fill(0);
  const biasedOutput =
    bias.length === multipliedFeature.length
      ? addVector(multipliedFeature, bias)
      : multipliedFeature;

  const multipliedX =
    currentNodeX + DISTANCE_TO_FEATURE * 2 + aggregatedFeature.length * cellWidth;
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
    currentNodeY - direction * DISTANCE_TO_FEATURE - FEATURE_HEIGHT / 2,
    bias,
    cellWidth
  );
  appendLine(
    group,
    multipliedX + multipliedFeature.length * cellWidth,
    currentNodeY,
    multipliedX + multipliedFeature.length * cellWidth + DISTANCE_TO_FEATURE,
    currentNodeY
  );
  appendPath(
    group,
    [
      [
        multipliedX + bias.length * cellWidth,
        currentNodeY - direction * DISTANCE_TO_FEATURE,
      ],
      [
        multipliedX + bias.length * cellWidth + DISTANCE_TO_FEATURE / 2,
        currentNodeY - direction * DISTANCE_TO_FEATURE,
      ],
      [
        multipliedX + bias.length * cellWidth + DISTANCE_TO_FEATURE / 2,
        currentNodeY,
      ],
      [multipliedX + bias.length * cellWidth + DISTANCE_TO_FEATURE, currentNodeY],
    ],
    { stroke: HIGHLIGHT, strokeWidth: 1.4 }
  );
  appendFeatureVector(
    group,
    multipliedX + DISTANCE_TO_FEATURE * 1.5 + multipliedFeature.length * cellWidth,
    currentNodeY - FEATURE_HEIGHT / 2,
    biasedOutput,
    cellWidth
  );
}

function drawExpansion(
  group: SVGGElement,
  target: Extract<StaticHoverTarget, { kind: "feature-node" }> | null,
  scene: StaticVisualizationScene,
  adjacencyMatrix: number[][],
  sortedGNNFeatures: number[][][],
  modelInfo: any,
  cellWidth: number
) {
  group.replaceChildren();
  if (!target) {
    return;
  }

  drawFeatureExpansion(
    group,
    target,
    scene,
    adjacencyMatrix,
    sortedGNNFeatures,
    modelInfo,
    cellWidth
  );
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

  const group = createSvgElement("g");
  group.classList.add("gnn-static-hover-marks");
  svg.append(group);
  const expansionGroup = createSvgElement("g");
  expansionGroup.classList.add("gnn-static-expansion-marks");
  svg.append(expansionGroup);
  wrapper.append(svg);
  const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
  let expandedTarget: Extract<StaticHoverTarget, { kind: "feature-node" }> | null = null;

  const onPointerMove = (event: PointerEvent) => {
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
    const point = getScenePoint(svg, scene, event);
    const target = hitTest(scene, point);
    if (target?.kind === "feature-node" && target.layerIndex > 0) {
      expandedTarget =
        expandedTarget?.layerIndex === target.layerIndex &&
        expandedTarget.nodeIndex === target.nodeIndex
          ? null
          : target;
      drawExpansion(
        expansionGroup,
        expandedTarget,
        scene,
        adjacencyMatrix,
        sortedGNNFeatures,
        modelInfo,
        cellWidth
      );
      return;
    }

    if (!target) {
      expandedTarget = null;
      expansionGroup.replaceChildren();
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
