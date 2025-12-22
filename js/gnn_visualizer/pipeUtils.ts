import * as d3 from "d3";

export function injectSVG(g:any, x: number, y: number, SVGPath:string, svgClass: string){
    d3.xml(SVGPath).then(function(data) {
        const svg = g!.node()!.appendChild(data.documentElement)
        d3.select(svg).attr("x", x).attr("y", y).attr("class", svgClass)
        return svg;
    });
}

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


type LayerData = number[][];
type ModelInfoRaw = Record<string, number[][]>;

export function extractSortedGNNLayerFeatures(modelInfo: ModelInfoRaw): LayerData[] {
    return Object.keys(modelInfo)
        .filter(key => key.startsWith("gnn_layer_"))
        .sort((a, b) => {
            const aIdx = parseInt(a.split("_")[2], 10);
            const bIdx = parseInt(b.split("_")[2], 10);
            return aIdx - bIdx;
        })
        .map(key => modelInfo[key].map(row => [...row] as number[]))

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

export function transitFeatureLayers(layerID: number, distanceX: number) {
    d3.selectAll(".feature-layer")
        .filter(function () {
            const idStr = (this as HTMLElement).id;
            const match = idStr.match(/^feature-layer-(\d+)-node-\d+$/);
            if (!match) return false;
            const id = Number(match[1]);
            return id >= layerID;
        }).transition().duration(500).ease(d3.easeCubicOut).attr("transform", `translate(${distanceX}, 0)`);
    transitFCLayer(distanceX);
}

export function transitFCLayer(distanceX: number) {
    d3.selectAll(".fc-feature-layer").transition().duration(500).ease(d3.easeCubicOut).attr("transform", `translate(${distanceX}, 0)`);
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

export function addVector(a: number[], b: number[]): number[] {
    if (a.length !== b.length) {
        throw new Error("Vector length mismatch");
    }
    return a.map((v, i) => v + b[i]);
}

export function scaleVector(k: number, v: number[]): number[] {
    return v.map(x => k * x);
}

export const countOnes = (arr: number[]) =>
    arr.reduce((sum, x) => sum + (x === 1 ? 1 : 0), 0);

export function vecMatMul(v: number[], W: number[][]): number[] {
    const n = v.length;
    const m = W[0].length;

    if (W.length !== n) {
        throw new Error("Dimension mismatch");
    }

    const result = Array(m).fill(0);

    for (let j = 0; j < m; j++) {
        for (let i = 0; i < n; i++) {
            result[j] += v[i] * W[i][j];
        }
    }

    return result;
}

export function randomMatrix(m: number, n: number): number[][] {
    return Array.from({ length: m }, () =>
        Array.from({ length: n }, () => Math.random())
    );
}

export function randomVector(n: number): number[] {
    return Array.from({ length: n }, () => Math.random());
}


export const featureColor = d3
    .scaleLinear<string>()
    .domain([-3, -1, -0.1, 0, 0.1, 1, 3])
    .range(["#304E30", "#3DBA41", "#B7EFB8", "white", "#BBB7EF", "#6E09CD", "#4B0092"]);

export const curve = d3.line().curve(d3.curveBasis);
