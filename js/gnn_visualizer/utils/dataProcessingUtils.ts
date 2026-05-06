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

export type ModelVisualizationData = {
    adjacencyMatrix: number[][];
    linkList: any[];
    queries: number[][];
    nodeLabels: string[];
    nodeOrder: number[];
    intmData: Record<string, any>;
    messagePassingDepth: number;
    isLocalEdgeView: boolean;
};

export type GraphAggregationInfo = {
    feature: number[];
    label: string;
};

const messagePassingLayerTypes = new Set(["GCNConv", "GATConv", "SAGEConv", "GraphSAGEConv"]);

export function getMessagePassingDepth(modelInfo: any, intmData: Record<string, number[][]>): number {
    const modelDepth = Object.values(modelInfo ?? {}).filter((layer: any) =>
        messagePassingLayerTypes.has(layer?.type)
    ).length;

    if (modelDepth > 0) {
        return modelDepth;
    }

    return Math.max(0, extractSortedGNNLayerFeatures(intmData).length - 1);
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

function isFiniteNumber(value: unknown): value is number {
    return typeof value === "number" && Number.isFinite(value);
}

function normalizeNumberVector(value: unknown): number[] | null {
    if (!Array.isArray(value) || value.length === 0 || !value.every(isFiniteNumber)) {
        return null;
    }

    return [...value];
}

function firstNumberVector(value: unknown, allowMultiRow: boolean = true): number[] | null {
    const vector = normalizeNumberVector(value);
    if (vector) {
        return vector;
    }

    if (!Array.isArray(value) || value.length === 0) {
        return null;
    }

    if (!allowMultiRow && value.length !== 1) {
        return null;
    }

    return normalizeNumberVector(value[0]);
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

function normalizeGraphAggregationLabel(value: unknown): string {
    if (typeof value !== "string" || value.trim() === "") {
        return "Pooling";
    }

    const normalized = value.trim().toLowerCase().replace(/[_-]/g, " ");
    if (normalized.includes("mean")) {
        return "Mean Pooling";
    }
    if (normalized.includes("add") || normalized.includes("sum")) {
        return "Sum Pooling";
    }
    if (normalized.includes("max")) {
        return "Max Pooling";
    }

    return value.trim().endsWith("Pooling") ? value.trim() : `${value.trim()} Pooling`;
}

function averageFeatures(features: number[][]): number[] | null {
    const validFeatures = features.filter((feature) => normalizeNumberVector(feature));
    const first = validFeatures[0];
    if (!first) {
        return null;
    }

    const result = Array(first.length).fill(0) as number[];
    for (const feature of validFeatures) {
        for (let index = 0; index < first.length; index += 1) {
            result[index] += feature[index] ?? 0;
        }
    }

    return result.map((value) => value / validFeatures.length);
}

function graphAggregationFromCandidate(
    candidate: unknown,
    fallbackLabel: unknown
): GraphAggregationInfo | null {
    if (isRecord(candidate)) {
        const feature =
            firstNumberVector(candidate.feature) ??
            firstNumberVector(candidate.vector) ??
            firstNumberVector(candidate.output) ??
            firstNumberVector(candidate.features);
        if (!feature) {
            return null;
        }

        return {
            feature,
            label: normalizeGraphAggregationLabel(
                candidate.type ?? candidate.aggregation ?? candidate.aggr ?? candidate.name ?? fallbackLabel
            ),
        };
    }

    const feature = firstNumberVector(candidate);
    if (!feature) {
        return null;
    }

    return {
        feature,
        label: normalizeGraphAggregationLabel(fallbackLabel),
    };
}

export function getGraphAggregationInfo(
    intmData: Record<string, unknown>,
    fallbackFeatures: number[][] = []
): GraphAggregationInfo | null {
    const candidates: Array<[string, unknown]> = [
        ["Pooling", intmData?.graphAggregation],
        ["Pooling", intmData?.graphPooling],
        ["Pooling", intmData?.pooling],
        ["Readout", intmData?.readout],
        ["Graph Embedding", intmData?.graphEmbedding],
        ["Graph Feature", intmData?.graphFeature],
        ["Graph Feature", intmData?.graph_feature],
        ["Graph Representation", intmData?.graph_repr],
    ];

    for (const [label, candidate] of candidates) {
        const aggregation = graphAggregationFromCandidate(candidate, label);
        if (aggregation) {
            return aggregation;
        }
    }

    const fallback = averageFeatures(fallbackFeatures);
    return fallback ? { feature: fallback, label: "Mean Pooling" } : null;
}

export function getGraphOutputFeature(intmData: Record<string, unknown>): number[] | null {
    const candidates = [
        intmData?.graphOutput,
        intmData?.graphPrediction,
        intmData?.graphLogits,
        intmData?.final,
        intmData?.softmax,
        intmData?.logits,
        intmData?.classifier,
        intmData?.modelOutput,
    ];

    for (const candidate of candidates) {
        const feature = firstNumberVector(candidate, false);
        if (feature) {
            return feature;
        }
    }

    return null;
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

function hasConnection(G: number[][], a: number, b: number) {
    return G[a]?.[b] !== 0 || G[b]?.[a] !== 0;
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
            if (hasConnection(G, node, v) && !visited.has(v)) {
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

function validQueriesForNodeCount(queries: number[][], nodeCount: number) {
    return queries.filter(([source, target]) =>
        Number.isInteger(source) &&
        Number.isInteger(target) &&
        source >= 0 &&
        target >= 0 &&
        source < nodeCount &&
        target < nodeCount
    );
}

function remapIntermediateData(intmData: Record<string, any>, nodeOrder: number[]) {
    const remapped: Record<string, any> = { ...intmData };

    for (const key of Object.keys(intmData ?? {})) {
        const layer = intmData[key];
        if (!key.startsWith("act") || !Array.isArray(layer)) {
            continue;
        }

        remapped[key] = nodeOrder.map((originalIndex) => {
            const feature = layer[originalIndex];
            return Array.isArray(feature) ? [...feature] : [];
        });
    }

    return remapped;
}

function remapLinks(linkList: any[], indexMap: Map<number, number>) {
    return linkList.reduce((result: any[], link) => {
        const source = indexMap.get(link.source);
        const target = indexMap.get(link.target);
        if (source == null || target == null) {
            return result;
        }

        result.push({ ...link, source, target });
        return result;
    }, []);
}

function remapQueries(queries: number[][], indexMap: Map<number, number>) {
    return queries.reduce((result: number[][], [source, target]) => {
        const localSource = indexMap.get(source);
        const localTarget = indexMap.get(target);
        if (localSource == null || localTarget == null) {
            return result;
        }

        result.push([localSource, localTarget]);
        return result;
    }, []);
}

export function getEdgeOutputFeature(
    intmData: Record<string, any>,
    sortedGNNFeatures: number[][][],
    queries: number[][],
    queryIndex: number
) {
    const decoderOutput = intmData?.decoder?.[queryIndex];
    if (Array.isArray(decoderOutput) && decoderOutput.length >= 2) {
        return decoderOutput.slice(0, 2);
    }

    const decoderProbability =
        Array.isArray(decoderOutput) && decoderOutput.length === 1
            ? decoderOutput[0]
            : typeof decoderOutput === "number"
                ? decoderOutput
                : null;

    if (typeof decoderProbability === "number" && Number.isFinite(decoderProbability)) {
        const probability = Math.max(0, Math.min(1, decoderProbability));
        return [1 - probability, probability];
    }

    const [source, target] = queries[queryIndex] ?? [];
    if (source == null || target == null) {
        return [0.5, 0.5];
    }

    const lastLayer = sortedGNNFeatures[sortedGNNFeatures.length - 1] ?? [];
    const sourceFeature = lastLayer[source];
    const targetFeature = lastLayer[target];
    if (!Array.isArray(sourceFeature) || !Array.isArray(targetFeature)) {
        return [0.5, 0.5];
    }

    const dot = sourceFeature.reduce(
        (sum, value, index) => sum + value * (targetFeature[index] ?? 0),
        0
    );
    const probability = 1 / (1 + Math.exp(-dot));
    return [1 - probability, probability];
}

export function buildModelVisualizationData(
    adjacencyMatrix: number[][],
    linkList: any[],
    intmData: Record<string, any>,
    modelInfo: any,
    queries: number[][],
    mode: string
): ModelVisualizationData {
    const messagePassingDepth = getMessagePassingDepth(modelInfo, intmData);
    const nodeCount = adjacencyMatrix.length;
    const fullNodeOrder = Array.from({ length: nodeCount }, (_, index) => index);
    const fullData: ModelVisualizationData = {
        adjacencyMatrix,
        linkList,
        queries,
        nodeLabels: fullNodeOrder.map(String),
        nodeOrder: fullNodeOrder,
        intmData,
        messagePassingDepth,
        isLocalEdgeView: false,
    };

    const edgeQueries = validQueriesForNodeCount(queries, nodeCount);
    if (mode !== "edge" || edgeQueries.length === 0) {
        return fullData;
    }

    const subgraph = processSubgraphDataPipe(
        adjacencyMatrix,
        edgeQueries,
        messagePassingDepth
    );
    if (subgraph.nodes.length === 0) {
        return fullData;
    }

    return {
        adjacencyMatrix: subgraph.subG,
        linkList: remapLinks(linkList, subgraph.indexMap),
        queries: remapQueries(edgeQueries, subgraph.indexMap),
        nodeLabels: subgraph.nodes.map(String),
        nodeOrder: subgraph.nodes,
        intmData: remapIntermediateData(intmData, subgraph.nodes),
        messagePassingDepth,
        isLocalEdgeView: true,
    };
}
