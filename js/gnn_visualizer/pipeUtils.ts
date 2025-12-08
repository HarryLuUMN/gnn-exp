
export function transformDataToMatrixVisFormat(nodes: any, links: any) {
    const matrixSize = nodes.length;
    let adjacancyMatrix: number[][] = Array.from({ length: matrixSize }, () =>
        Array(matrixSize).fill(0)
    );

    for(let i = 0; i < links.length; i++) {
        const link = links[i];
        const linkSource = link.source;
        const linkTarget = link.target;
        adjacancyMatrix[linkSource][linkTarget] = 1
    }

    for(let i = 0; i < matrixSize; i++) {
        for(let j = 0; j < matrixSize; j++) {
            if(i==j)adjacancyMatrix[i][j] = 1;
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

