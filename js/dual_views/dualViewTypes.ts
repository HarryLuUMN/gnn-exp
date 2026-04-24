export type NodeDatum = {
  id: number;
  element: string;
  community?: number;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
};

export type LinkDatum = {
  source: number;
  target: number;
  attr?: any;
};

export type HoverState =
  | { kind: "node"; nodeId: number }
  | { kind: "edge"; a: number; b: number }
  | null;
