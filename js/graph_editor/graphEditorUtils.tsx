export async function loadSimGraphData(dataFile: string) {
    const response = await fetch(dataFile);
    const jsonData = await response.json();

    if (!jsonData.x || !jsonData.edge_index) {
        throw new Error("Invalid graph JSON: missing fields");
    }

    const ret = {
        x: jsonData.x,
        edge_index: jsonData.edge_index,
        edge_attr: jsonData.edge_attr ?? [],
        y: jsonData.y ?? [],
        batch: jsonData.batch ?? [],
    };

    console.log("Loaded graph data - from utils:", ret);

    return ret;
}

export function randomizeFeatures(dim: number) {
    const randomFeature = [];
    for (let i = 0; i < dim; i++) {
        randomFeature.push((Math.random() * 3 - 1.5).toFixed(4));
    }

    const featureText = randomFeature.join(", ");
    return featureText;
}

export function parseFeatureText(text: string): number[] {
    return text
        .split(",")
        .map((v) => parseFloat(v.trim()))
        .filter((v) => !isNaN(v));
}

export function processDataFromVisualizerToEditor(input: {
  x: number[][],
  edge_index: number[][],
  edge_attr: number[][],
  y: number[],
  batch: number[]
}) {
  const { x, edge_index } = input;
  const edge_attr = input.edge_attr ?? [];

  // IMPORTANT: keep original features
  const nodes = x.map((features, index) => ({
    id: `N${index}`,
    group: 1,
    feature: [...features],   // ⭐ Restore the original feature!
  }));

  const links = edge_index[0].map((sourceIdx, i) => {
    const targetIdx = edge_index[1][i];
    return {
      source: `N${sourceIdx}`,
      target: `N${targetIdx}`,
      value: edge_attr[i]?.[0] ?? 1
    };
  });

  return { nodes, links };
}


export function processDataFromEditorToVisualizer(input: {
  nodes: { id: string; group: number; feature?: number[] }[];
  links: { source: any; target: any; value: number }[];
}) {
  const edgeIndexPair1: number[] = [];
  const edgeIndexPair2: number[] = [];
  const pairSet = new Set<string>();
  const edgeAttr: number[][] = [];
  const nodeIndexById = new Map(input.nodes.map((node, index) => [node.id, index]));

  const resolveNodeIndex = (endpoint: any) => {
    if (typeof endpoint === "number") return endpoint;
    if (typeof endpoint === "string") return nodeIndexById.get(endpoint);
    if (endpoint && typeof endpoint.index === "number") return endpoint.index;
    if (endpoint && typeof endpoint.id === "string") return nodeIndexById.get(endpoint.id);
    return undefined;
  };

  // === Build edge_index & edge_attr ===
  for (let i = 0; i < input.links.length; i++) {
    const link = input.links[i];
    const source = resolveNodeIndex(link.source);
    const target = resolveNodeIndex(link.target);

    if (source == null || target == null) {
      continue;
    }

    const key1 = `${source},${target}`;
    const key2 = `${target},${source}`;

    if (!pairSet.has(key1)) {
      edgeIndexPair1.push(source);
      edgeIndexPair2.push(target);
      edgeAttr.push([link.value]);
      pairSet.add(key1);
    }

    if (!pairSet.has(key2)) {
      edgeIndexPair1.push(target);
      edgeIndexPair2.push(source);
      edgeAttr.push([link.value]);
      pairSet.add(key2);
    }
  }

  // === Build x from node.feature ===
  const node_num = input.nodes.length;
  const x: number[][] = [];

  for (let i = 0; i < node_num; i++) {
    const node = input.nodes[i];

    if (node.feature && Array.isArray(node.feature)) {
      x.push(node.feature);
    } else {
      // If missing feature → fallback to zeros
      const dim = input.nodes[0]?.feature?.length || 5;
      x.push(Array(dim).fill(0));
    }
  }

  // Labels y & batch (placeholder)
  const y = [0];
  const batch = Array(node_num).fill(0);

  return {
    edge_index: [edgeIndexPair1, edgeIndexPair2],
    edge_attr: edgeAttr,
    x,
    y,
    batch,
  };
}
