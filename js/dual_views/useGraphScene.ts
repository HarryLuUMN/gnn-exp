import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
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
  const simulationRef = useRef<d3.Simulation<SceneNode, LinkDatum> | null>(null);
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

      simulationRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    const previousNodes = new Map(
      nodesRef.current.map((node) => [node.id, node] as const)
    );
    const centerX = (width - 2 * padding) / 2;
    const centerY = (height - 2 * padding) / 2;

    nodesRef.current = nodes.map((node, index) => {
      const previous = previousNodes.get(node.id);
      return {
        ...node,
        x:
          previous?.x ??
          centerX + Math.cos(index) * 10 + (Math.random() - 0.5) * 10,
        y:
          previous?.y ??
          centerY + Math.sin(index) * 10 + (Math.random() - 0.5) * 10,
        vx: previous?.vx ?? 0,
        vy: previous?.vy ?? 0,
        fx: null,
        fy: null,
      };
    });

    linksRef.current = links.map((link) => ({ ...link }));
    const simulationLinks = links.map((link) => ({ ...link }));

    simulationRef.current?.stop();

    if (nodesRef.current.length === 0) {
      scheduleUpdate();
      return;
    }

    const simulation = d3
      .forceSimulation(nodesRef.current)
      .force(
        "link",
        d3
          .forceLink<SceneNode, LinkDatum>(simulationLinks)
          .id((datum) => datum.id)
          .distance(80)
      )
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(centerX, centerY))
      .force("x", d3.forceX(centerX).strength(0.1))
      .force("y", d3.forceY(centerY).strength(0.1))
      .on("tick", scheduleUpdate);

    simulationRef.current = simulation;
    scheduleUpdate();

    return () => {
      simulation.stop();
      if (simulationRef.current === simulation) {
        simulationRef.current = null;
      }
    };
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
      node.fx = node.x ?? 0;
      node.fy = node.y ?? 0;
      simulationRef.current?.alphaTarget(0.3).restart();
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

      node.fx = x;
      node.fy = y;
      scheduleUpdate();
    },
    [scheduleUpdate]
  );

  const endDrag = useCallback(() => {
    const node = draggingNodeRef.current;
    if (!node) {
      return;
    }

    node.fx = null;
    node.fy = null;
    draggingNodeRef.current = null;
    simulationRef.current?.alphaTarget(0);
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
