function isRecord(value: unknown): value is Record<string, unknown> {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === "number" && Number.isFinite(value);
}

function normalizeMatrix(value: unknown, inputLength: number): number[][] | null {
    if (!Array.isArray(value) || value.length === 0) {
        return null;
    }

    const rows = value.map((row) => {
        if (!Array.isArray(row) || row.length !== inputLength || !row.every(isFiniteNumber)) {
            return null;
        }
        return [...row];
    });

    if (rows.some((row) => row === null)) {
        return null;
    }

    return rows as number[][];
}

export function normalizeLayerWeightMatrix(
    layerInfo: unknown,
    inputLength: number
): number[][] | null {
    if (!isRecord(layerInfo) || inputLength === 0) {
        return null;
    }

    const rows = normalizeMatrix(layerInfo.weight, inputLength);
    if (!rows) {
        return null;
    }

    return rows[0].map((_, columnIndex) => rows.map((row) => row[columnIndex]));
}

export function normalizeLayerBias(
    layerInfo: unknown,
    outputLength: number
): number[] {
    if (isRecord(layerInfo) && Array.isArray(layerInfo.bias)) {
        const bias = layerInfo.bias;
        if (bias.length === outputLength && bias.every(isFiniteNumber)) {
            return [...bias];
        }
    }

    return Array(Math.max(0, outputLength)).fill(0) as number[];
}
