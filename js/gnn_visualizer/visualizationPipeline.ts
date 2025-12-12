import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import * as d3 from "d3";
import { curve, extractSortedGNNLayerFeatures, featureColor, removeRepeatLinks, transformDataToMatrixVisFormat } from "./pipeUtils";
import { interactFCNodesAndLinksPipe, interactNodesAndLinksPipe } from "./interactionPipeline";

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
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(modelInfo);
    
    console.log("Processed nodes and links:", nodeList, linkList);
    
    // visualization pipes
    visualizeMatrixPipe(adjacancyMatrix);
    visualizeIntermediateFeaturePipe(intmData, modelInfo, adjacancyMatrix);
    visualizeLinksBetweenLayersPipe(linkList, 100, modelInfo);

    // interaction pipes
    interactNodesAndLinksPipe(adjacancyMatrix);
    interactFCNodesAndLinksPipe(sortedGNNFeatures, adjacancyMatrix);

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
        svg.append("rect")
            .attr("x", startX)
            .attr("y", startY+ i * cellSize)
            .attr("width", cellSize * adjacencyMatrix.length)
            .attr("height", cellSize)
            .attr("fill", "none")
            .attr("class", "adj-matrix-row-border")
            .attr("id", `adj-matrix-row-border-${i}`)
            .style("stroke", "black")
            .style("stroke-width", 1)
            .style("opacity", 0);

        svg.append("rect")
            .attr("x", startX+ i * cellSize)
            .attr("y", startY)
            .attr("width", cellSize )
            .attr("height", cellSize* adjacencyMatrix.length)
            .attr("fill", "none")
            .attr("class", "adj-matrix-col-border")
            .attr("id", `adj-matrix-col-border-${i}`)
            .style("stroke", "black")
            .style("stroke-width", 1)
            .style("opacity", 0);
    }
    svg.selectAll(".adj-matrix-row-border, .adj-matrix-col-border").raise();
    
}

export function visualizeIntermediateFeaturePipe(intmData: any, modelInfo: any, adjacencyMatrix: number[][]){
    console.log("inside visualizeIntermediateFeaturePipe", intmData, adjacencyMatrix);

    const cellWidth = 6;
    const cellHeight = 12;
    const gapXBetweenLayers = 100;

    const svg = d3.select('#matrix-svg');
    const startX = adjacencyMatrix.length * 20 + 20 + 50;
    const startY = 50;

    console.log("modelInfo", modelInfo); 
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(modelInfo);
    console.log("Sorted GNN Layer Features:", sortedGNNFeatures);

    let layerX = startX;
    for(let i=0; i < sortedGNNFeatures.length; i++){
        
        if (i > 0)layerX += cellWidth * sortedGNNFeatures[i-1][0].length;
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
                    .style("stroke-width", 0.5)
                    .style("stroke", "gray")
                    .style("stroke-opacity", 0.5)
                    .style("opacity", 1);
            }
        }
        layerX +=  (gapXBetweenLayers);
    }
    visualizeFCFeaturesPipe(layerX, modelInfo);
}

export function visualizeLinksBetweenLayersPipe(
    links: any,
    gapSize: number,
    modelInfo: any
){
    console.log("start visualizeLinksBetweenLayers");
    // visualize links between GNN layers
    const svg = d3.select('#matrix-svg');

    console.log("modelInfo inside visualizeLinksBetweenLayers:", modelInfo);
    console.log("sortedGNNFeatures:", extractSortedGNNLayerFeatures(modelInfo));


    const sortedGNNFeatures = extractSortedGNNLayerFeatures(modelInfo);
    // const undirectLinks = removeRepeatLinks(links);

    const startX = 50 + sortedGNNFeatures[0].length * 20 + 20;
    const startY = 50;

    let layerX = startX + sortedGNNFeatures[0][0].length * 6;

    // looping through layers
    for(let i=0; i < sortedGNNFeatures.length - 1; i++){
        // compute locations
        const prevLayerX = layerX;
        layerX += sortedGNNFeatures[i+1][0].length * 6 + gapSize;
        const midLayerX = (prevLayerX + layerX) / 2;
        // looping through nodes in layer i
        for (let j = 0; j < links.length; j++) {
            const link = links[j];
            const sourceIdx = link.source;
            const targetIdx = link.target;

            const sourceY = startY + sourceIdx * 20 + 12;
            const targetY = startY + targetIdx * 20 + 12;

            const pathStart: [number, number] = [prevLayerX, sourceY];
            const pathEnd: [number, number] = [layerX, targetY];

            const mid1: [number, number] = [midLayerX, sourceY];
            const mid2: [number, number] = [midLayerX, targetY];
            svg.append("path")
                .attr("d", curve([pathStart, mid1, mid2, pathEnd]))
                .attr("stroke", "black")
                .attr("opacity", 0.1)
                .attr("fill", "none")
                .attr("class", "link-path")
                .attr("id", `link-path-${i}-${sourceIdx}-to-${targetIdx}`)
                .lower();
        }
        // visualize self-looping
        for(let n=0; n < sortedGNNFeatures[i].length; n++){
            const layerY = startY + n * 20 + 12;
            svg.append("line")
                .attr("x1", layerX)
                .attr("y1", layerY)
                .attr("x2", layerX - 125)
                .attr("y2", layerY)
                .attr("stroke", "black")
                .attr("opacity", 0.1)
                .attr("fill", "none")
                .attr("class", "link-path")
                .attr("id", `link-path-${i}-${n}-to-${n}`)
                .lower();
        }
    }
}

export function visualizeFCFeaturesPipe(layerX: number, modelInfo: any){
    console.log("inside visualizeFCFeaturesPipe", modelInfo);
    // get the last layer number from modelInfo
    const sortedLayers = extractSortedGNNLayerFeatures(modelInfo);
    const lastLayerNum = sortedLayers[sortedLayers.length - 1].length;
    const fcLayerFeatures: any[][] = modelInfo[`fc_layer_1`];
    console.log("fc data", fcLayerFeatures, lastLayerNum);
    const layerY = 50;
    const svg = d3.select('#matrix-svg');
    for(let i=0; i < fcLayerFeatures.length; i++){
        const feature: any[] | undefined = fcLayerFeatures[i];
        console.log("fc feature:", feature, i);
        if (!Array.isArray(feature)) continue;
        const g = svg.append("g").attr("class", "fc-feature-layer").attr("id", `fc-feature-layer-node-${i}`);

        g.append("rect")
            .attr("x", layerX)
            .attr("y", layerY + i * 20 + 6)
            .attr("width", feature.length * 6)
            .attr("height", 12)
            .attr("fill", "none")
            .attr("class", "fc-feature-layer-frame")
            .attr("id", `fc-feature-layer-frame-node-${i}`)
            .style("stroke-width", 2)
            .style("stroke", "black")
            .style("opacity", 0.5);

        for(let j=0; j < feature.length; j++){
            g.append("rect")
                .attr("x", layerX + j * 6)
                .attr("y", layerY + i * 20 + 6)
                .attr("width", 6)
                .attr("height", 12)
                .attr("fill", featureColor(feature[j]))
                .attr("class", "fc-feature-cell")
                .attr("id", `fc-feature-layer-node-${i}-dim-${j}`)
                .style("stroke-width", 0.5)
                .style("stroke", "gray")
                .style("stroke-opacity", 0.5)
                .style("opacity", 1);
        }

        

        svg.append("line")
            .attr("x1", layerX)
            .attr("y1", layerY + i * 20 + 12)
            .attr("x2", layerX - 100)
            .attr("y2", layerY + i * 20 + 12)
            .attr("stroke", "black")
            .attr("opacity", 0.1)
            .attr("fill", "none")
            .attr("class", "link-path-fc")
            .attr("id", `link-path-fc-${i}`)
            .lower();
    }
}
