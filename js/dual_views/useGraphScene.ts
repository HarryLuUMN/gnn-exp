import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { UndirectedGraph } from "graphology";
import louvain from "graphology-communities-louvain";
import forceAtlas2 from "graphology-layout-forceatlas2";
import noverlap from "graphology-layout-noverlap";
import type { LinkDatum, NodeDatum } from "./dualViewTypes";
import type { SceneNode } from "./renderers/shared";

interface UseGraphSceneArgs {
  nodes: NodeDatum[];
  links: LinkDatum[];
  width: number;
  height: number;
  padding: number;
  linkPredictionMode: boolean;
  onNodePositionChange?: (positions: { id: number; x: number; y: number }[]) => void;
}

type LayoutNodeAttributes = {
  id: number;
  x: number;
  y: number;
  size: number;
  community?: number;
};

type LayoutEdgeAttributes = {
  weight: number;
};

const NODE_LAYOUT_SIZE = 24;
const FIT_MARGIN = 36;
const MIN_SPAN = 1e-6;

function buildLayoutGraph(
  nodes: NodeDatum[],
  links: LinkDatum[],
  width: number,
  height: number,
  padding: number
) {
  const graph = new UndirectedGraph<LayoutNodeAttributes, LayoutEdgeAttributes>();
  const innerWidth = width - 2 * padding;
  const innerHeight = height - 2 * padding;
  const radius = Math.max(Math.min(innerWidth, innerHeight) * 0.35, 1);
  const centerX = innerWidth / 2;
  const centerY = innerHeight / 2;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  nodes.forEach((node, index) => {
    const angle = index * goldenAngle;
    const ring = radius * Math.sqrt((index + 0.5) / Math.max(nodes.length, 1));

    graph.addNode(String(node.id), {
      id: node.id,
      x: node.x ?? centerX + Math.cos(angle) * ring,
      y: node.y ?? centerY + Math.sin(angle) * ring,
      size: NODE_LAYOUT_SIZE,
    });
  });

  links.forEach((link) => {
    if (link.source === link.target) {
      return;
    }

    const source = String(link.source);
    const target = String(link.target);

    if (!graph.hasNode(source) || !graph.hasNode(target)) {
      return;
    }

    const [edge, edgeWasAdded] = graph.mergeEdge(source, target, { weight: 1 });
    if (!edgeWasAdded) {
      graph.updateEdgeAttribute(edge, "weight", (weight = 0) => weight + 1);
    }
  });

  return graph;
}

function seedCommunities(
  graph: UndirectedGraph<LayoutNodeAttributes, LayoutEdgeAttributes>,
  communities: Record<string, number>
) {
  const grouped = new Map<number, string[]>();

  graph.forEachNode((key) => {
    const community = communities[key] ?? 0;
    const members = grouped.get(community);
    if (members) {
      members.push(key);
    } else {
      grouped.set(community, [key]);
    }
    graph.setNodeAttribute(key, "community", community);
  });

  const communityIds = Array.from(grouped.keys()).sort((a, b) => a - b);
  const outerRadius = Math.max(communityIds.length * NODE_LAYOUT_SIZE * 1.5, 120);

  communityIds.forEach((community, communityIndex) => {
    const members = grouped.get(community) ?? [];
    const communityAngle = (communityIndex / Math.max(communityIds.length, 1)) * Math.PI * 2;
    const communityX = Math.cos(communityAngle) * outerRadius;
    const communityY = Math.sin(communityAngle) * outerRadius;
    const memberRadius = Math.max(Math.sqrt(members.length) * NODE_LAYOUT_SIZE, NODE_LAYOUT_SIZE);

    members.forEach((key, memberIndex) => {
      const angle = (memberIndex / Math.max(members.length, 1)) * Math.PI * 2;
      graph.mergeNodeAttributes(key, {
        x: communityX + Math.cos(angle) * memberRadius,
        y: communityY + Math.sin(angle) * memberRadius,
      });
    });
  });
}

function getForceAtlasIterations(order: number, size: number) {
  if (order <= 80) {
    return 160;
  }

  if (size > order * 80) {
    return 70;
  }

  return order > 600 ? 90 : 120;
}

function fitGraphToViewport(
  graph: UndirectedGraph<LayoutNodeAttributes, LayoutEdgeAttributes>,
  nodes: NodeDatum[],
  width: number,
  height: number,
  padding: number
): SceneNode[] {
  const innerWidth = width - 2 * padding;
  const innerHeight = height - 2 * padding;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  graph.forEachNode((_key, attr) => {
    minX = Math.min(minX, attr.x);
    maxX = Math.max(maxX, attr.x);
    minY = Math.min(minY, attr.y);
    maxY = Math.max(maxY, attr.y);
  });

  const spanX = Math.max(maxX - minX, MIN_SPAN);
  const spanY = Math.max(maxY - minY, MIN_SPAN);
  const fitWidth = Math.max(innerWidth - FIT_MARGIN * 2, 1);
  const fitHeight = Math.max(innerHeight - FIT_MARGIN * 2, 1);
  const scale = Math.min(fitWidth / spanX, fitHeight / spanY);
  const offsetX = FIT_MARGIN + (fitWidth - spanX * scale) / 2;
  const offsetY = FIT_MARGIN + (fitHeight - spanY * scale) / 2;

  return nodes.map((node) => {
    const attr = graph.getNodeAttributes(String(node.id));
    return {
      ...node,
      community: attr.community,
      x: offsetX + (attr.x - minX) * scale,
      y: offsetY + (attr.y - minY) * scale,
      fx: null,
      fy: null,
    };
  });
}

export function layoutGraph(
  nodes: NodeDatum[],
  links: LinkDatum[],
  width: number,
  height: number,
  padding: number
): SceneNode[] {
  const graph = buildLayoutGraph(nodes, links, width, height, padding);

  if (graph.order === 0) {
    return [];
  }

  const communities =
    graph.size > 0
      ? louvain(graph, {
          getEdgeWeight: "weight",
          randomWalk: false,
        })
      : Object.fromEntries(graph.nodes().map((key, index) => [key, index]));
  seedCommunities(graph, communities);

  if (graph.size > 0) {
    const inferredSettings = forceAtlas2.inferSettings(graph);
    forceAtlas2.assign(graph, {
      iterations: getForceAtlasIterations(graph.order, graph.size),
      getEdgeWeight: "weight",
      settings: {
        ...inferredSettings,
        barnesHutOptimize: graph.order > 120,
        barnesHutTheta: 0.7,
        linLogMode: true,
        outboundAttractionDistribution: true,
        gravity: graph.order > 400 ? 1.8 : 1.2,
        scalingRatio: graph.order > 400 ? 12 : 8,
        slowDown: graph.order > 400 ? 8 : 4,
      },
    });
  }

  noverlap.assign(graph, {
    maxIterations: graph.order > 400 ? 80 : 60,
    settings: {
      gridSize: Math.max(20, Math.ceil(Math.sqrt(graph.order))),
      margin: 4,
      ratio: 1.15,
      speed: 3,
    },
  });

  return fitGraphToViewport(graph, nodes, width, height, padding);
}

export function useGraphScene({
  nodes,
  links,
  width,
  height,
  padding,
  linkPredictionMode,
  onNodePositionChange,
}: UseGraphSceneArgs) {
  const nodesRef = useRef<SceneNode[]>([]);
  const linksRef = useRef<LinkDatum[]>([]);
  const draggingNodeRef = useRef<SceneNode | null>(null);
  const frameRef = useRef<number | null>(null);
  const [sceneVersion, setSceneVersion] = useState(0);

  const scheduleUpdate = useCallback(() => {
    if (frameRef.current !== null) {
      return;
    }

    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = null;
      setSceneVersion((value) => value + 1);
    });
  }, []);

  useEffect(() => {
    return () => {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    linksRef.current = links.map((link) => ({ ...link }));
    nodesRef.current = layoutGraph(nodes, links, width, height, padding);
    scheduleUpdate();
  }, [height, links, nodes, padding, scheduleUpdate, width]);

  const emitPositions = useCallback(() => {
    if (linkPredictionMode || !onNodePositionChange) {
      return;
    }

    onNodePositionChange(
      nodesRef.current.map((node) => ({
        id: node.id,
        x: node.x ?? 0,
        y: node.y ?? 0,
      }))
    );
  }, [linkPredictionMode, onNodePositionChange]);

  const beginDrag = useCallback(
    (nodeId: number) => {
      const node = nodesRef.current.find((item) => item.id === nodeId);
      if (!node) {
        return false;
      }

      draggingNodeRef.current = node;
      scheduleUpdate();
      return true;
    },
    [scheduleUpdate]
  );

  const dragTo = useCallback(
    (x: number, y: number) => {
      const node = draggingNodeRef.current;
      if (!node) {
        return;
      }

      node.x = x;
      node.y = y;
      scheduleUpdate();
    },
    [scheduleUpdate]
  );

  const endDrag = useCallback(() => {
    if (!draggingNodeRef.current) {
      return;
    }

    draggingNodeRef.current = null;
    scheduleUpdate();
    emitPositions();
  }, [emitPositions, scheduleUpdate]);

  const nodesById = useMemo(
    () => new Map(nodesRef.current.map((node) => [node.id, node] as const)),
    [sceneVersion]
  );

  return {
    nodes: nodesRef.current,
    links: linksRef.current,
    nodesById,
    sceneVersion,
    beginDrag,
    dragTo,
    endDrag,
  };
}
