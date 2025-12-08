import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import * as d3 from "d3";
import { transformDataToMatrixVisFormat } from "./pipeUtils";

export function visualizationPipeline(
    setIsLoading:any, 
    modelInfo:any, 
    intmData:any, 
    graphData:any
){
    console.log("Starting visualization pipeline...", modelInfo, intmData, graphData, setIsLoading);
    const { nodeList, linkList} = preMatrixVisualizationDataProcessingPipe("node prediction", undefined, undefined, graphData);
    const adjacancyMatrix = transformDataToMatrixVisFormat(nodeList, linkList);
    console.log("Processed nodes and links:", nodeList, linkList);
    visualizeMatrixPipe(adjacancyMatrix);
    visualizeIntermediateFeaturePipe(intmData, adjacancyMatrix);
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

export function visualizeIntermediateFeaturePipe(intmData: any, adjacencyMatrix: number[][]){
    console.log("inside visualizeIntermediateFeaturePipe", intmData, adjacencyMatrix);

    const cellWidth = 2;
    const cellHeight = 12;
    const gapBetweenLayers = 25;

    const svg = d3.select('#matrix-svg');
    const startX = 50 + adjacencyMatrix.length * 20 + 20;
    const startY = 50;



}

export function visualizeFeature(
    feature: number[], 
    cellWidth: number, 
    cellHeight: number, 
    startX: number, 
    startY: number, 
    layerIndex: number
){

}

export function visualizeLinksBetweenLayers(
    adjacencyMatrix: number[][],
    gapSize: number,
){

}

export function visualizeLinksForAggregation(){

}


