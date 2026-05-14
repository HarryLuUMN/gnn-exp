import { countOnes } from "./mathUtils";

export type AggregationKind =
    | "gcn-normalized"
    | "sum"
    | "mean"
    | "max"
    | "min"
    | "median"
    | "std"
    | "var";

export type AggregationContribution = {
    nodeIndex: number;
    label: string;
    value?: number;
};

export type AggregationResult = {
    kind: AggregationKind;
    label: string;
    aggregatedFeature: number[];
    contributions: AggregationContribution[];
};

type NeighborFeature = {
    nodeIndex: number;
    feature: number[];
};

const warnedUnsupportedAggregations = new Set<string>();

function isRecord(value: unknown): value is Record<string, unknown> {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

function stringifyAggregation(value: unknown): string | null {
    if (typeof value === "string") {
        return value;
    }

    if (Array.isArray(value)) {
        const names = value
            .map((item) => stringifyAggregation(item))
            .filter((item): item is string => item !== null);
        return names.length > 0 ? names.join(",") : null;
    }

    if (isRecord(value)) {
        const nested = stringifyAggregation(value.aggregation ?? value.aggr ?? value.name);
        if (nested) {
            return nested;
        }

        const ctor = value.constructor;
        if (typeof ctor === "function" && typeof ctor.name === "string") {
            return ctor.name;
        }
    }

    return null;
}

export function normalizeAggregationKind(value: unknown): AggregationKind | null {
    const raw = stringifyAggregation(value);
    if (!raw) {
        return null;
    }

    let normalized = raw.trim().toLowerCase().replace(/_/g, "-").replace(/\s+/g, "");
    if (normalized.endsWith("aggregation")) {
        normalized = normalized.slice(0, -"aggregation".length);
    }

    switch (normalized) {
        case "gcn":
        case "gcnconv":
        case "gcn-normalized":
        case "gcn-normalised":
        case "normalized-gcn":
        case "normalised-gcn":
            return "gcn-normalized";
        case "add":
        case "sum":
        case "gat":
        case "gatconv":
        case "attention":
        case "attention-basic":
            return "sum";
        case "gin":
        case "ginconv":
            return "sum";
        case "sage":
        case "sageconv":
        case "graphsage":
        case "graphsageconv":
            return "mean";
        case "avg":
        case "average":
        case "mean":
            return "mean";
        case "maximum":
        case "max":
            return "max";
        case "minimum":
        case "min":
            return "min";
        case "median":
            return "median";
        case "stdev":
        case "std":
            return "std";
        case "variance":
        case "var":
            return "var";
        default:
            return null;
    }
}

function warnUnsupportedAggregation(value: unknown) {
    const label = stringifyAggregation(value) ?? String(value);
    if (warnedUnsupportedAggregations.has(label)) {
        return;
    }

    warnedUnsupportedAggregations.add(label);
    console.warn(
        `Unsupported GNN aggregation "${label}". Falling back to gcn-normalized.`
    );
}

export function resolveLayerAggregation(layerInfo: unknown): AggregationKind {
    if (!isRecord(layerInfo)) {
        return "gcn-normalized";
    }

    const explicitAggregation = layerInfo.aggregation ?? layerInfo.aggr;
    const explicitKind = normalizeAggregationKind(explicitAggregation);
    if (explicitKind) {
        return explicitKind;
    }

    if (explicitAggregation != null) {
        warnUnsupportedAggregation(explicitAggregation);
        return "gcn-normalized";
    }

    if (normalizeAggregationKind(layerInfo.type) === "gcn-normalized") {
        return "gcn-normalized";
    }

    return "gcn-normalized";
}

export function aggregationDisplayLabel(kind: AggregationKind) {
    return kind === "gcn-normalized" ? "GCN norm" : kind;
}

function firstFeatureLength(features: number[][]) {
    const firstFeature = features.find((feature) => Array.isArray(feature));
    return firstFeature?.length ?? 0;
}

function zeroFeature(length: number) {
    return Array(Math.max(0, length)).fill(0) as number[];
}

function normalizeFeature(feature: unknown, length: number): number[] | null {
    if (!Array.isArray(feature) || feature.length !== length) {
        return null;
    }

    return feature.map((value) =>
        typeof value === "number" && Number.isFinite(value) ? value : 0
    );
}

function collectNeighborFeatures(
    adjacencyMatrix: number[][],
    previousLayer: number[][],
    nodeIndex: number
) {
    const featureLength = firstFeatureLength(previousLayer);
    const adjacencyRow = adjacencyMatrix[nodeIndex] ?? [];
    const neighbors: NeighborFeature[] = [];

    if (featureLength === 0) {
        return { featureLength, neighbors };
    }

    for (let index = 0; index < adjacencyRow.length; index += 1) {
        if (adjacencyRow[index] !== 1) {
            continue;
        }

        const feature = normalizeFeature(previousLayer[index], featureLength);
        if (feature) {
            neighbors.push({ nodeIndex: index, feature });
        }
    }

    return { featureLength, neighbors };
}

function featureWiseMedian(values: number[]) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];
}

function aggregateFeatureWise(
    neighbors: NeighborFeature[],
    featureLength: number,
    kind: Exclude<AggregationKind, "gcn-normalized">
) {
    if (neighbors.length === 0) {
        return zeroFeature(featureLength);
    }

    const result: number[] = zeroFeature(featureLength);
    for (let dim = 0; dim < featureLength; dim += 1) {
        const values = neighbors.map(({ feature }) => feature[dim]);
        if (kind === "sum") {
            result[dim] = values.reduce((sum, value) => sum + value, 0);
        } else if (kind === "mean") {
            result[dim] =
                values.reduce((sum, value) => sum + value, 0) / values.length;
        } else if (kind === "max") {
            result[dim] = Math.max(...values);
        } else if (kind === "min") {
            result[dim] = Math.min(...values);
        } else if (kind === "median") {
            result[dim] = featureWiseMedian(values);
        } else {
            const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
            const variance =
                values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
                values.length;
            result[dim] = kind === "std" ? Math.sqrt(variance) : variance;
        }
    }

    return result;
}

function contributionLabel(kind: AggregationKind, count: number, value?: number) {
    if (kind === "gcn-normalized") {
        return typeof value === "number" ? value.toFixed(2) : "gcn";
    }

    if (kind === "mean") {
        return count > 0 ? (1 / count).toFixed(2) : "mean";
    }

    return kind;
}

export function aggregateNeighborFeatures(
    adjacencyMatrix: number[][],
    sortedGNNFeatures: number[][][],
    layerIndex: number,
    nodeIndex: number,
    layerInfo: unknown
): AggregationResult {
    const previousLayer = sortedGNNFeatures[layerIndex - 1] ?? [];
    const { featureLength, neighbors } = collectNeighborFeatures(
        adjacencyMatrix,
        previousLayer,
        nodeIndex
    );
    const kind = resolveLayerAggregation(layerInfo);

    if (kind !== "gcn-normalized") {
        return {
            kind,
            label: aggregationDisplayLabel(kind),
            aggregatedFeature: aggregateFeatureWise(neighbors, featureLength, kind),
            contributions: neighbors.map(({ nodeIndex: neighborIndex }) => ({
                nodeIndex: neighborIndex,
                label: contributionLabel(kind, neighbors.length),
            })),
        };
    }

    const aggregatedFeature = zeroFeature(featureLength);
    const targetDegree = Math.max(1, countOnes(adjacencyMatrix[nodeIndex] ?? []));
    const contributions: AggregationContribution[] = [];

    for (const { nodeIndex: neighborIndex, feature } of neighbors) {
        const sourceDegree = Math.max(
            1,
            countOnes(adjacencyMatrix[neighborIndex] ?? [])
        );
        const value = 1 / Math.sqrt(targetDegree * sourceDegree);
        for (let dim = 0; dim < featureLength; dim += 1) {
            aggregatedFeature[dim] += value * feature[dim];
        }
        contributions.push({
            nodeIndex: neighborIndex,
            value,
            label: contributionLabel(kind, neighbors.length, value),
        });
    }

    return {
        kind,
        label: aggregationDisplayLabel(kind),
        aggregatedFeature,
        contributions,
    };
}
