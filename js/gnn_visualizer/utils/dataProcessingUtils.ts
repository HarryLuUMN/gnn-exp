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
