import * as d3 from "d3";

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

export function getMaxLayerID(): number {
    let maxLayerID = -Infinity;

    d3.selectAll(".feature-layer").each(function () {
        const idStr = (this as HTMLElement).id;
        const match = idStr.match(/^feature-layer-(\d+)-node-\d+$/);
        if (!match) return;

        const layerID = Number(match[1]);
        if (layerID > maxLayerID) {
            maxLayerID = layerID;
        }
    });

    return maxLayerID;
}

export function extractFCNodeIndex(id: string): number {
    const match = id.match(/^fc-feature-layer-node-(\d+)$/);
    if (!match) return -1;
    return Number(match[1]);
}

type SubgraphResult = {
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
        if (dist === d) continue;

        for (let v = 0; v < N; v++) {
            if (G[node][v] !== 0 && !visited.has(v)) {
                visited.add(v);
                queue.push({ node: v, dist: dist + 1 });
            }
        }
    }

    // local index -> global node id
    const nodes = Array.from(visited);

    // global node id -> local index
    const indexMap = new Map<number, number>();
    nodes.forEach((nodeId, localIdx) => {
        indexMap.set(nodeId, localIdx);
    });

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
    const mergedNodeSet = new Set<number>();
    g1.nodes.forEach(n => mergedNodeSet.add(n));
    g2.nodes.forEach(n => mergedNodeSet.add(n));

    const nodes = Array.from(mergedNodeSet);

    const indexMap = new Map<number, number>();
    nodes.forEach((nodeId, localIdx) => {
        indexMap.set(nodeId, localIdx);
    });

    const M = nodes.length;
    const subG = Array.from({ length: M }, () => Array(M).fill(0));

    for (let i = 0; i < g1.nodes.length; i++) {
        for (let j = 0; j < g1.nodes.length; j++) {
            const w = g1.subG[i][j];
            if (w !== 0) {
                const uGlobal = g1.nodes[i];
                const vGlobal = g1.nodes[j];
                const uLocal = indexMap.get(uGlobal)!;
                const vLocal = indexMap.get(vGlobal)!;
                subG[uLocal][vLocal] = w;
            }
        }
    }

    for (let i = 0; i < g2.nodes.length; i++) {
        for (let j = 0; j < g2.nodes.length; j++) {
            const w = g2.subG[i][j];
            if (w !== 0) {
                const uGlobal = g2.nodes[i];
                const vGlobal = g2.nodes[j];
                const uLocal = indexMap.get(uGlobal)!;
                const vLocal = indexMap.get(vGlobal)!;
                subG[uLocal][vLocal] = w;
            }
        }
    }

    return { subG, nodes, indexMap };
}

export function flattenAndUnique(arr: number[][]): number[] {
    return Array.from(new Set(arr.flat()));
}
  
export function subgraphDataProcessingPipe(adjacencyMatrix: number[][], queries: number[][], distance: number){
    const flattenedQueries = flattenAndUnique(queries);
    let G: SubgraphResult = {subG: [], nodes: [], indexMap: new Map<number, number>()};
    for (let i = 0; i < flattenedQueries.length; i++){
        let g = extractKHopSubgraph(adjacencyMatrix, flattenedQueries[i], distance);
        G = mergeSubgraphs(G, g);
    }
    console.log("G in subgraphDataProcessingPipe:", G);
    return G;
}