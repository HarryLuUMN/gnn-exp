import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import * as d3 from "d3";
import { transformDataToMatrixVisFormat } from "./pipeUtils";

export function visualizationPipeline(
    setIsLoading:any, 
    graph_path:string, 
    intmData:any, 
    graphData:any
){
    console.log("Starting visualization pipeline...", graph_path, intmData, graphData, setIsLoading);
    const { nodeList, linkList} = preMatrixVisualizationDataProcessingPipe("node prediction", undefined, undefined, graphData);
    console.log("Processed nodes and links:", nodeList, linkList);
    visualizeMatrixPipe(nodeList, linkList);
    return null;
}

export function visualizeMatrixPipe(nodes: any, links: any) {

    console.log("inside visualizeMatrixPipe", nodes, links);

    const graphSize = nodes.length;
    const adjacancyMatrix = transformDataToMatrixVisFormat(nodes, links);
    console.log("Adjacency Matrix in the Vis Pipe:", adjacancyMatrix);

    const g = d3.select("#matvis");
    g.selectAll("*").remove();

    const width = 800;
    const height = 800;

    const svg = g.append("svg").attr("width", width).attr("height", height);

    // visualize the matrix
    const startX = 50;
    const startY = 50;
    const cellSize = 20;

    for(let i = 0; i < adjacancyMatrix.length; i++) {
        for(let j = 0; j < adjacancyMatrix[i].length; j++) {
            if(adjacancyMatrix[i][j] === 1) {
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
    // fill: rgb(105, 179, 162);fill: rgb(238, 238, 238);

    
}
