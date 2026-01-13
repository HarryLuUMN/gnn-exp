export function computeFeatureLayerX(
    startX: number,
    layerID: number,
    cellWidth: number,
    gapSize: number,
    sortedGNNFeatures: any[][],
): number {
    console.log("computeFeatureLayerX", startX, layerID, cellWidth, gapSize, sortedGNNFeatures);   
    let x = startX;
    for (let i = 0; i < layerID; i++) {
        const featureDim = sortedGNNFeatures[i][0].length;
        x += featureDim * cellWidth;
        if(i!=0){
            x += gapSize;
           //  x += sortedGNNFeatures[i][0].length * cellWidth; 
        }
    }
    return x+2;
}

export function computeFeatureLayerY(
    nodeID: number,
    startY: number,
    nodeHeight: number
): number {
    return startY + nodeID * nodeHeight + 12;
}

