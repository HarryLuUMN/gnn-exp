import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import * as d3 from "d3";
import { extractSortedGNNLayerFeatures, featureColor, transformDataToMatrixVisFormat } from "./pipeUtils";

export function visualizationPipeline(
    setIsLoading:any, 
    modelInfo:any, 
    intmData:any, 
    graphData:any
){
    console.log("Starting visualization pipeline...", modelInfo, intmData, graphData, setIsLoading);
    
    // data processing pipes
    const { nodeList, linkList} = preMatrixVisualizationDataProcessingPipe("node prediction", undefined, undefined, graphData);
    const adjacancyMatrix = transformDataToMatrixVisFormat(nodeList, linkList);
    
    console.log("Processed nodes and links:", nodeList, linkList);
    
    // visualization pipes
    visualizeMatrixPipe(adjacancyMatrix);
    visualizeIntermediateFeaturePipe(intmData, modelInfo, adjacancyMatrix);
    visualizeLinksBetweenLayers(adjacancyMatrix, 100);
    
    return null;
}

export function visualizeMatrixPipe(adjacencyMatrix: number[][]) {
    
    console.log("Adjacency Matrix in the Vis Pipe:", adjacencyMatrix);

    const g = d3.select("#matvis");
    g.selectAll("*").remove();

    const width = 2400;
    const height = 800;

    const svg = g
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                .attr("id", "matrix-svg");

    // visualize the matrix
    const startX = 50;
    const startY = 50;
    const cellSize = 20;

    for(let i = 0; i < adjacencyMatrix.length; i++) {
        for(let j = 0; j < adjacencyMatrix[i].length; j++) {
            if(adjacencyMatrix[i][j] === 1) {
                svg.append("rect")
                    .attr("x", startX + j * cellSize)
                    .attr("y", startY + i * cellSize)
                    .attr("width", cellSize)
                    .attr("height", cellSize)
                    .attr("fill", "rgb(105, 179, 162)")
                    .attr("class", "adj-matrix-cell")
                    .attr("id", `cell-${i}-${j}`)
                    .style("stroke", "white")
                    .style("stroke-width", 1)
                    .style("opacity", 0.8);
            } else {
                svg.append("rect")
                    .attr("x", startX + j * cellSize)
                    .attr("y", startY + i * cellSize)
                    .attr("width", cellSize)
                    .attr("height", cellSize)
                    .attr("fill", "rgb(238, 238, 238)")
                    .attr("class", "adj-matrix-cell")
                    .attr("id", `cell-${i}-${j}`)
                    .style("stroke", "white")
                    .style("stroke-width", 1)
                    .style("opacity", 0.8);
            }
        }
    }
    
}

export function visualizeIntermediateFeaturePipe(intmData: any, modelInfo: any, adjacencyMatrix: number[][]){
    console.log("inside visualizeIntermediateFeaturePipe", intmData, adjacencyMatrix);

    const cellWidth = 6;
    const cellHeight = 12;
    const gapXBetweenLayers = 100;

    const svg = d3.select('#matrix-svg');
    const startX = 50 + adjacencyMatrix.length * 20 + 20;
    const startY = 50;

    console.log("modelInfo", modelInfo); 
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(modelInfo);
    console.log("Sorted GNN Layer Features:", sortedGNNFeatures);

    for(let i=0; i < sortedGNNFeatures.length; i++){
        const layerX = startX + i * (cellWidth + gapXBetweenLayers);
        const layerFeatures = sortedGNNFeatures[i];
        for(let j=0; j < layerFeatures.length; j++){
            const layerY = startY + j * (20);
            const feature = layerFeatures[j];
            const g = svg.append("g").attr("class", "feature-layer").attr("id", `feature-layer-${i}-node-${j}`);
            
            g.append("rect")
                .attr("x", layerX)
                .attr("y", layerY + (cellHeight/2))
                .attr("width", feature.length * cellWidth)
                .attr("height", cellHeight)
                .attr("fill", "none")
                .attr("class", "feature-layer-frame")
                .attr("id", `feature-layer-frame-${i}-node-${j}`)
                .style("stroke-width", 2)
                .style("stroke", "black")
                .style("opacity", 0.5);

            for(let k=0; k < feature.length; k++){
                g.append("rect")
                    .attr("x", layerX + k * cellWidth)
                    .attr("y", layerY + (cellHeight/2))
                    .attr("width", cellWidth)
                    .attr("height", cellHeight)
                    .attr("fill", featureColor(feature[k]))
                    .attr("class", "feature-cell")
                    .attr("id", `feature-layer-${i}-node-${j}-dim-${k}`)
                    .style("stroke-width", 1)
                    .style("stroke", "gray")
                    .style("stroke", "none")
                    .style("opacity", 0.9);
            }
        }
    }

}

export function visualizeLinksBetweenLayers(
    adjacencyMatrix: number[][],
    gapSize: number,
){
    // visualize links between GNN layers
    
}


