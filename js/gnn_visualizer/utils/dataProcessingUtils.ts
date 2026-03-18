export function transformDataToMatrixVisFormat(nodes: any, links: any) {
    const matrixSize = nodes.length;
    let adjacancyMatrix: number[][] = Array.from({ length: matrixSize }, () =>
        Array(matrixSize).fill(0)
    );

    for (let i = 0; i < links.length; i++) {
        const link = links[i];
        const linkSource = link.source;
        const linkTarget = link.target;
        adjacancyMatrix[linkSource][linkTarget] = 1
    }

    for (let i = 0; i < matrixSize; i++) {
        for (let j = 0; j < matrixSize; j++) {
            if (i == j) adjacancyMatrix[i][j] = 1;
        }
    }

    console.log("adjacancyMatrix:", adjacancyMatrix);

    return adjacancyMatrix;
}

export function extractSortedGNNLayerFeatures(intmData: Record<string, number[][]>): number[][][] {

    console.log("intmData in extractSortedGNNLayerFeatures:", intmData);
    const sortedGNNLayerFeatures = Object.keys(intmData)
        .filter(key => key.startsWith("act"))
        .sort((a, b) => {
            const aIdx = parseInt(a.replace("act", ""), 10)
            const bIdx = parseInt(b.replace("act", ""), 10)
            return aIdx - bIdx
        })
        .map(key => intmData[key].map(row => [...row]));

    console.log("sortedGNNLayerFeatures:", sortedGNNLayerFeatures);

    return sortedGNNLayerFeatures;
}

export function removeRepeatLinks(links: any[]) {
    const seen = new Set<string>();
    const result: any[] = [];

    for (const link of links) {
        const { source, target } = link;
        const key = source < target
            ? `${source}-${target}`
            : `${target}-${source}`;

        if (!seen.has(key)) {
            seen.add(key);
            result.push(link);
        }
    }

    return result;
}

export function extractFeatureId(id: any) {
    return id.match(/^feature-layer-(\d+)-node-(\d+)$/);
}

export function extractFCNodeIndex(id: string): number {
    const match = id.match(/^fc-feature-layer-node-(\d+)$/);
    if (!match) return -1;
    return Number(match[1]);
}

export type SubgraphResult = {
    subG: number[][];
    nodes: number[];                
    indexMap: Map<number, number>;   
};

export function extractKHopSubgraph(
    G: number[][],
    center: number,
    d: number
): SubgraphResult {
    const N = G.length;
    const visited = new Set<number>();
    const queue: Array<{ node: number; dist: number }> = [];

    visited.add(center);
    queue.push({ node: center, dist: 0 });

    while (queue.length > 0) {
        const { node, dist } = queue.shift()!;
        if (dist >= d) continue;

        for (let v = 0; v < N; v++) {
            if (G[node][v] !== 0 && !visited.has(v)) {
                visited.add(v);
                queue.push({ node: v, dist: dist + 1 });
            }
        }
    }

    const nodes = Array.from(visited).sort((a, b) => a - b);

    const indexMap = new Map<number, number>();
    nodes.forEach((id, idx) => indexMap.set(id, idx));

    const M = nodes.length;
    const subG = Array.from({ length: M }, () => Array(M).fill(0));

    for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
            subG[i][j] = G[nodes[i]][nodes[j]];
        }
    }

    return { subG, nodes, indexMap };
}

export function mergeSubgraphs(
    g1: SubgraphResult,
    g2: SubgraphResult
): SubgraphResult {
    if (g1.nodes.length === 0) return g2;
    if (g2.nodes.length === 0) return g1;

    const mergedNodeSet = new Set<number>();
    g1.nodes.forEach(n => mergedNodeSet.add(n));
    g2.nodes.forEach(n => mergedNodeSet.add(n));

    const nodes = Array.from(mergedNodeSet).sort((a, b) => a - b);

    const indexMap = new Map<number, number>();
    nodes.forEach((id, idx) => indexMap.set(id, idx));

    const M = nodes.length;
    const subG = Array.from({ length: M }, () => Array(M).fill(0));

    for (let i = 0; i < g1.nodes.length; i++) {
        for (let j = 0; j < g1.nodes.length; j++) {
            const w = g1.subG[i][j];
            if (w !== 0) {
                const u = indexMap.get(g1.nodes[i])!;
                const v = indexMap.get(g1.nodes[j])!;
                subG[u][v] = w;
            }
        }
    }

    for (let i = 0; i < g2.nodes.length; i++) {
        for (let j = 0; j < g2.nodes.length; j++) {
            const w = g2.subG[i][j];
            if (w !== 0) {
                const u = indexMap.get(g2.nodes[i])!;
                const v = indexMap.get(g2.nodes[j])!;
                subG[u][v] = w;
            }
        }
    }

    return { subG, nodes, indexMap };
}

export function flattenAndUnique(arr: number[][]): number[] {
    return Array.from(new Set(arr.flat()));
}

export function processSubgraphDataPipe(
    adjacencyMatrix: number[][],
    queries: number[][],
    distance: number
): SubgraphResult {
    const centers = flattenAndUnique(queries);

    let G: SubgraphResult = {
        subG: [],
        nodes: [],
        indexMap: new Map<number, number>()
    };

    for (const c of centers) {
        const g = extractKHopSubgraph(adjacencyMatrix, c, distance);
        G = mergeSubgraphs(G, g);
    }

    return G;
}

export function processSubgraphSequenceDataPipe(
    adjacencyMatrix: number[][],
    queries: number[][],
    distance: number
): SubgraphResult[] {
    const subgraphs: SubgraphResult[] = [];
    for (let i = distance; i >= 0; i--) {
        const subgraph = processSubgraphDataPipe(adjacencyMatrix, queries, i);
        subgraphs.push(subgraph);
    }
    return subgraphs;   
}

export function getVisibleNodeIdsForLayer(
    layerFeatures: any[][],
    subgraphData: SubgraphResult[],
    layerIndex: number,
    subgraphSample: any
): number[] {
    if (!subgraphSample) {
        return layerFeatures.map((_, index) => index);
    }

    const subgraph = subgraphData[layerIndex];
    if (!subgraph) {
        return layerFeatures.map((_, index) => index);
    }

    return subgraph.nodes.filter((nodeId) => Array.isArray(layerFeatures[nodeId]));
}

export function buildVisibleLayerNodeIds(
    sortedGNNFeatures: any[][][],
    subgraphData: SubgraphResult[],
    subgraphSample: any
): number[][] {
    return sortedGNNFeatures.map((layerFeatures, layerIndex) =>
        getVisibleNodeIdsForLayer(layerFeatures, subgraphData, layerIndex, subgraphSample)
    );
}
