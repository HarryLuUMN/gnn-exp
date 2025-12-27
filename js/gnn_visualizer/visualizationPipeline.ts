import * as d3 from "d3";
import { addVector, countOnes, curve, extractSortedGNNLayerFeatures, featureColor, injectSVG, randomMatrix, randomVector, scaleVector, vecMatMul } from "./pipeUtils";
import { computeFeatureLayerX, computeFeatureLayerY } from "./geometryUtils";
import { distanceToFeature } from "../utils/const";

export function visualizationPipeline(cellWidth: number, cellHeight: number, adjacencyMatrix: number[][], intmData: any, linkList: any[]) {
    // define parameters
    const gapSizeBetweenLayers = 100;
    
    // visualization pipes
    visualizeMatrixPipe(adjacencyMatrix);
    visualizeIntermediateFeaturePipe(cellWidth, cellHeight, gapSizeBetweenLayers, intmData, adjacencyMatrix);
    visualizeLinksBetweenLayersPipe(linkList, gapSizeBetweenLayers, intmData);
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

export function visualizeIntermediateFeaturePipe(cellWidth: number, cellHeight: number, gapXBetweenLayers: number, intmData: any, adjacencyMatrix: number[][]){
    console.log("inside visualizeIntermediateFeaturePipe", intmData, adjacencyMatrix);

    const svg = d3.select('#matrix-svg');
    const startX = adjacencyMatrix.length * 20 + 20 + 50;
    const startY = 50;

    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
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
    visualizeFCForEachSingleNodeSubpipe(layerX, intmData);
}

export function visualizeLinksBetweenLayersPipe(
    links: any,
    gapSize: number,
    intmData: any
){
    console.log("start visualizeLinksBetweenLayers");
    // visualize links between GNN layers
    const svg = d3.select('#matrix-svg');

    console.log("intmData inside visualizeLinksBetweenLayers:", intmData);
    console.log("sortedGNNFeatures:", extractSortedGNNLayerFeatures(intmData));


    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
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

export function visualizeFCForEachSingleNodeSubpipe(layerX: number, intmData: any){
    console.log("inside visualizeFCFeaturesPipe", intmData);
    // get the last layer number from intmData
    const sortedLayers = extractSortedGNNLayerFeatures(intmData);
    const lastLayerNum = sortedLayers[sortedLayers.length - 1].length;
    const fcLayerFeatures: any[][] = intmData[`softmax`];
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

export function visualizeInnerGNNLayerSubpipe(cellWidth: number, layerID: number, nodeID: number, adjacencyMatrix: number[][], sortedGNNFeatures: any[][], modelInfo: any, direction: string){
    console.log("inside layer modelInfo:", modelInfo);
    const distanceBetweenFeatures = 50;
    const gapXBetweenLayers = 100;
    const startX = adjacencyMatrix.length * 20 + 20 + 50;
    const g = d3.select('#matrix-svg');
    const inner = g.append("g").attr("class", "layer-inner-works-group").attr("id", `layer-inner-works-group-layer-${layerID}-node-${nodeID}`);

    const currentNodeX = computeFeatureLayerX(startX, layerID, cellWidth, gapXBetweenLayers, sortedGNNFeatures);
    const currentNodeY = computeFeatureLayerY(nodeID, 50, 20);

    let locations = [];

    let aggregatedFeature: number[] = Array(sortedGNNFeatures[layerID-1][0].length).fill(0);
    let degreeMultipliers: number[] = [];

    let dirCoefficient = 1;
    if (direction === "up") dirCoefficient = -1;

    for(let j = 0; j < adjacencyMatrix[nodeID].length; j++) {
        if (adjacencyMatrix[nodeID][j] === 1){
            const targetNodeX = computeFeatureLayerX(startX, layerID, cellWidth, gapXBetweenLayers, sortedGNNFeatures);
            const targetNodeY = computeFeatureLayerY(j, 50, 20);
            locations.push([targetNodeX, targetNodeY]);
            const degreeMultiplier = 1 / (Math.sqrt(countOnes(adjacencyMatrix[nodeID]) * countOnes(adjacencyMatrix[j])));
            aggregatedFeature = addVector(aggregatedFeature, scaleVector(degreeMultiplier, sortedGNNFeatures[layerID-1][j]));
            degreeMultipliers.push(degreeMultiplier);
        }
    }
    console.log("aggregatedFeature", aggregatedFeature);
    const firstIntersect: [number, number] = [currentNodeX + distanceBetweenFeatures, currentNodeY];
    // visualize aggregated links
    const ctrlPointForCurrentNode: [number, number] = [currentNodeX + (distanceBetweenFeatures / 2), currentNodeY];
    for(let k = 0; k < locations.length; k++){
        const ctrlPointForTargetNode: [number, number] = [currentNodeX + (distanceBetweenFeatures / 2), locations[k][1]];
        const path = inner.append("path")
            .attr("d", curve([[locations[k][0], locations[k][1]], ctrlPointForTargetNode, ctrlPointForCurrentNode, firstIntersect]))
            .attr("stroke", "black")
            .attr("opacity", 1)
            .attr("fill", "none")
            .attr("class", "link-path-aggregated layer-inner-works")
            .attr("id", `link-path-aggregated-${layerID}-${nodeID}-to-${k}`)
            .lower();
        inner.append("text")
            .attr("x", locations[k][0] + 3)
            .attr("y", locations[k][1] - 6)
            .text(degreeMultipliers[k].toFixed(2))
            .attr("class", "degree-multiplier-text layer-inner-works")
            .attr("id", `degree-multiplier-text-${layerID}-${nodeID}-to-${k}`)
            .style("font-size", "6px")
            .lower();
    }
    // visualize aggregated feature
    const aggregatedFeatureGroup = inner.append("g").attr("class", "aggregated-feature-layer layer-inner-works").attr("id", `aggregated-feature-layer-layer-${layerID}-node-${nodeID}`);
    aggregatedFeatureGroup.append("rect")
        .attr("x", currentNodeX + distanceBetweenFeatures)
        .attr("y", currentNodeY - 12/2)
        .attr("width", aggregatedFeature.length * cellWidth)
        .attr("height", 12)
        .attr("fill", "none")
        .attr("class", "aggregated-feature-frame layer-inner-works")
        .attr("id", `aggregated-feature-frame-layer-${layerID}-node-${nodeID}`)
        .style("stroke-width", 1)
        .style("stroke", "black")
        .style("opacity", 1);
    for(let l=0; l < aggregatedFeature.length; l++){
        aggregatedFeatureGroup.append("rect")
            .attr("x", currentNodeX + distanceBetweenFeatures + l * cellWidth)
            .attr("y", currentNodeY - 12/2)
            .attr("width", cellWidth)
            .attr("height", 12)
            .attr("fill", featureColor(aggregatedFeature[l]))
            .attr("class", "aggregated-feature-cell layer-inner-works")
            .attr("id", `aggregated-feature-layer-node-${layerID}-node-${nodeID}-dim-${l}`)
            .style("stroke-width", 0.5)
            .style("stroke", "gray")
            .style("stroke-opacity", 0.5)
            .style("opacity", 1).lower();
    }
    // visualize weight matrix and it multiplication with aggregated feature
    inner.append("line")
        .attr("x1", currentNodeX + distanceBetweenFeatures + aggregatedFeature.length * cellWidth)
        .attr("y1", currentNodeY)
        .attr("x2", currentNodeX + distanceBetweenFeatures*2 + aggregatedFeature.length * cellWidth)
        .attr("y2", currentNodeY)
        .attr("stroke", "black")
        .attr("opacity", 1)
        .attr("fill", "none")
        .attr("class", "aggregated-feature-to-multiplied-feature-line layer-inner-works")
    // add matmul icon
    injectSVG(inner, currentNodeX + distanceBetweenFeatures*1.5 + aggregatedFeature.length * cellWidth, currentNodeY, "./assets/matmul.svg", "matmul-icon layer-inner-works");
    inner.append("path")
        .attr("d", curve([
            [currentNodeX + distanceBetweenFeatures*1.5 + aggregatedFeature.length * cellWidth, currentNodeY], 
            [currentNodeX + distanceBetweenFeatures*1.5 + aggregatedFeature.length * cellWidth, currentNodeY + (dirCoefficient) * distanceBetweenFeatures * 0.5],
            [currentNodeX + distanceBetweenFeatures*1.5 + aggregatedFeature.length * cellWidth - distanceBetweenFeatures * 0.5, currentNodeY + (dirCoefficient) * distanceBetweenFeatures * 0.5],
            [currentNodeX + distanceBetweenFeatures*1.5 + aggregatedFeature.length * cellWidth - distanceBetweenFeatures * 0.5, currentNodeY + (dirCoefficient) * distanceBetweenFeatures]
        ]))
        .attr("stroke", "black")
        .attr("opacity", 1)
        .attr("fill", "none")
        .attr("class", "weight-matrix-to-intersect-path layer-inner-works")
        .lower();
    const weightMatrix = randomMatrix(sortedGNNFeatures[layerID-1][0].length, sortedGNNFeatures[layerID][0].length);
    const matrixStartX = currentNodeX + distanceBetweenFeatures*1.5 + aggregatedFeature.length * cellWidth - distanceBetweenFeatures*0.5 - cellWidth * weightMatrix[0].length / 2;
    const matrixStartY = currentNodeY + (dirCoefficient) * distanceBetweenFeatures * 1;
    let matrixStartYOffset = 0;
    if (direction === "up") matrixStartYOffset = -cellWidth * (weightMatrix.length - 1);
    inner.append("rect")
        .attr("x", matrixStartX)
        .attr("y", matrixStartY + matrixStartYOffset)
        .attr("width", cellWidth * weightMatrix[0].length)
        .attr("height", cellWidth * weightMatrix.length)
        .attr("fill", "none")
        .attr("class", "weight-matrix-frame layer-inner-works")
        .attr("id", `weight-matrix-frame-${layerID}-${nodeID}`)
        .style("stroke-width", 1)
        .style("stroke", "black")
        .style("opacity", 1)
        .lower();
    for(let m=0; m < weightMatrix.length; m++){
        for(let n=0; n < weightMatrix[m].length; n++){
            inner.append("rect")
                .attr("x", matrixStartX + n * cellWidth)
                .attr("y", matrixStartY + (dirCoefficient) * m * cellWidth)
                .attr("width", cellWidth)
                .attr("height", cellWidth)
                .attr("fill", featureColor(weightMatrix[m][n]))
                .attr("class", "weight-matrix-cell layer-inner-works")
                .attr("id", `weight-matrix-cell-${layerID}-${nodeID}-dim-${m}-${n}`)
                .style("opacity", 1);
        }
    }
    // visualize bias and actiivation function
    const multipliedFeature = vecMatMul(aggregatedFeature, weightMatrix);
    const bias = randomVector(sortedGNNFeatures[layerID][0].length);
    inner.append("rect")
        .attr("x", currentNodeX + distanceBetweenFeatures*2 + aggregatedFeature.length * cellWidth)
        .attr("y", currentNodeY - 12/2)
        .attr("width", multipliedFeature.length * cellWidth)
        .attr("height", 12)
        .attr("fill", "none")
        .attr("class", "multiplied-feature-frame layer-inner-works")
        .attr("id", `multiplied-feature-frame-${layerID}-${nodeID}`)
        .style("stroke-width", 1)
        .style("stroke", "black")
        .style("opacity", 1)
        .lower();
    for (let m=0; m < multipliedFeature.length; m++){
        inner.append("rect")
            .attr("x", currentNodeX + distanceBetweenFeatures*2 + aggregatedFeature.length * cellWidth + m * cellWidth)
            .attr("y", currentNodeY - 12/2)
            .attr("width", cellWidth)
            .attr("height", 12)
            .attr("fill", featureColor(multipliedFeature[m]))
            .attr("class", "multiplied-feature-cell layer-inner-works")
            .attr("id", `multiplied-feature-cell-${layerID}-${nodeID}-dim-${m}`)
            .style("opacity", 1)
            .lower();
    }
    inner.append("rect")
        .attr("x", currentNodeX + distanceBetweenFeatures*2 + aggregatedFeature.length * cellWidth)
        .attr("y", currentNodeY - (dirCoefficient) * distanceBetweenFeatures)
        .attr("width", bias.length * cellWidth)
        .attr("height", 12)
        .attr("fill", "none")
        .attr("class", "bias-frame layer-inner-works")
        .attr("id", `bias-frame-${layerID}-${nodeID}`)
        .style("stroke-width", 1)
        .style("stroke", "black")
        .style("opacity", 1)
        .lower();
    for (let m=0; m < bias.length; m++){
        inner.append("rect")
            .attr("x", currentNodeX + distanceBetweenFeatures*2 + aggregatedFeature.length * cellWidth + m * cellWidth)
            .attr("y", currentNodeY - (dirCoefficient) * distanceBetweenFeatures)
            .attr("width", cellWidth)
            .attr("height", 12)
            .attr("fill", featureColor(bias[m]))
            .attr("class", "bias-cell layer-inner-works")
            .attr("id", `bias-cell-${layerID}-${nodeID}-dim-${m}`)
            .style("opacity", 1)
            .lower();
    }
    inner.append("line")
        .attr("x1", currentNodeX + distanceBetweenFeatures*2 + aggregatedFeature.length * cellWidth + multipliedFeature.length * cellWidth)
        .attr("y1", currentNodeY)
        .attr("x2", currentNodeX + distanceBetweenFeatures*3 + aggregatedFeature.length * cellWidth + multipliedFeature.length * cellWidth)
        .attr("y2", currentNodeY)
        .attr("stroke", "black")
        .attr("opacity", 1)
        .attr("fill", "none")
        .attr("class", "multiplied-feature-to-bias-line layer-inner-works")
        .lower();
    inner.append("path")
        .attr("d", curve([
            [currentNodeX + distanceBetweenFeatures*2 + aggregatedFeature.length * cellWidth + bias.length * cellWidth, currentNodeY - (dirCoefficient) * distanceBetweenFeatures + 12/2], 
            [currentNodeX + distanceBetweenFeatures*2.5 + aggregatedFeature.length * cellWidth + bias.length * cellWidth, currentNodeY - (dirCoefficient) * distanceBetweenFeatures + 12/2],
            [currentNodeX + distanceBetweenFeatures*2.5 + aggregatedFeature.length * cellWidth + bias.length * cellWidth, currentNodeY],
            [currentNodeX + distanceBetweenFeatures*3 + aggregatedFeature.length * cellWidth + bias.length * cellWidth, currentNodeY]
        ]))
        .attr("stroke", "black")
        .attr("opacity", 1)
        .attr("fill", "none")
        .attr("class", "bias-to-output-path layer-inner-works")
        .lower();
    // add activation function icon
    
}

export function visualizeInnerFCLayerSubpipe(cellWidth: number, nodeID: number, sortedGNNFeatures: any[][], direction: string){
    const startX = sortedGNNFeatures[0].length * 20 + 20 + 50;

    const weightMatrix = randomMatrix(2, 4);
    const bias = randomVector(4);
    const currentNodeX = computeFeatureLayerX(startX, sortedGNNFeatures.length, cellWidth, 100, sortedGNNFeatures);
    const currentNodeY = computeFeatureLayerY(nodeID, 50, 20);

    const inner = d3.select('#matrix-svg').append("g").attr("class", "layer-inner-works-group");

    let dirCoefficient = 1;
    if (direction === "up") dirCoefficient = -1;

    // input to weighted vector path
    inner.append("line")
        .attr("x1", currentNodeX)
        .attr("y1", currentNodeY)
        .attr("x2", currentNodeX + distanceToFeature)
        .attr("y2", currentNodeY)
        .attr("stroke", "black")
        .attr("opacity", 1)
        .attr("class", "weight-line layer-inner-works")
        .lower();
    // intersect to weight matrix path
    const matrixStartX = currentNodeX + distanceToFeature;
    const matrixStartY = currentNodeY - (dirCoefficient) * distanceToFeature;
    inner.append("path")
        .attr("d", curve([
            [currentNodeX + distanceToFeature / 2, currentNodeY],
            [currentNodeX + distanceToFeature / 2, currentNodeY - distanceToFeature / 2 * (dirCoefficient)],
            [matrixStartX + cellWidth * weightMatrix[0].length / 2, currentNodeY - distanceToFeature / 2 * (dirCoefficient)],
            [matrixStartX + cellWidth * weightMatrix[0].length / 2, matrixStartY]
        ]))
        .attr("stroke", "black")
        .attr("opacity", 1)
        .attr("class", "intersect-to-weight-matrix-path layer-inner-works")
        .style("fill", "none")
        .lower();
    // visualize weight matrix
    inner.append("rect")
        .attr("x", matrixStartX)
        .attr("y", matrixStartY)
        .attr("width", cellWidth * weightMatrix[0].length)
        .attr("height", cellWidth * weightMatrix.length)
        .attr("fill", "none")
        .attr("class", "weight-matrix-frame layer-inner-works")
        .attr("id", `fc-weight-matrix-frame-node-${nodeID}`)
        .style("stroke-width", 1)
        .style("stroke", "black")
        .style("opacity", 1)
        .lower();
    for(let m=0; m < weightMatrix.length; m++){
        for(let n=0; n < weightMatrix[m].length; n++){
            inner.append("rect")
                .attr("x", matrixStartX + n * cellWidth)
                .attr("y", matrixStartY + m * cellWidth)
                .attr("width", cellWidth)
                .attr("height", cellWidth)
                .attr("fill", featureColor(weightMatrix[m][n]))
                .attr("class", "weight-matrix-cell layer-inner-works")
                .attr("id", `fc-weight-matrix-cell-node-${nodeID}-dim-${m}-${n}`)
                .style("opacity", 1);
        }
    }
    // visualize multiplied vector
    const multipliedFeature = vecMatMul(sortedGNNFeatures[sortedGNNFeatures.length -1][nodeID], weightMatrix);
    inner.append("rect").attr("x", currentNodeX + distanceToFeature).attr("y", currentNodeY - 12/2).attr("width", multipliedFeature.length * cellWidth).attr("height", 12).attr("fill", "none").attr("class", "multiplied-feature-frame layer-inner-works").attr("id", `fc-multiplied-feature-frame-node-${nodeID}`).style("stroke-width", 1).style("stroke", "black").style("opacity", 1)
    for(let l=0; l < multipliedFeature.length; l++) inner.append("rect").attr("x", currentNodeX + distanceToFeature + l * cellWidth).attr("y", currentNodeY - 12/2).attr("width", cellWidth).attr("height", 12).attr("fill", featureColor(multipliedFeature[l])).lower();
    // visualize bias vector and its addition
    inner.append("line")
        .attr("x1", currentNodeX + distanceToFeature + multipliedFeature.length * cellWidth)
        .attr("y1", currentNodeY)
        .attr("x2", currentNodeX + distanceToFeature * 2 + multipliedFeature.length * cellWidth)
        .attr("y2", currentNodeY)
        .attr("stroke", "black")
        .attr("opacity", 1)
        .attr("class", "multiplied-to-bias-line layer-inner-works")
        .lower();
    inner.append("path")
        .attr("d", curve([
            [currentNodeX + distanceToFeature + multipliedFeature.length * cellWidth, currentNodeY + (dirCoefficient) * distanceToFeature / 2],
            [currentNodeX + distanceToFeature * 1.5 + multipliedFeature.length * cellWidth, currentNodeY + (dirCoefficient) * distanceToFeature / 2],
            [currentNodeX + distanceToFeature * 1.5 + multipliedFeature.length * cellWidth, currentNodeY],
            [currentNodeX + distanceToFeature * 2 + multipliedFeature.length * cellWidth, currentNodeY],
        ])).attr("stroke", "black").attr("opacity", 1).attr("fill", "none").attr("class", "bias-to-output-path layer-inner-works")
    // bias vector
    for(let m=0; m < bias.length; m++){
        inner.append("rect")
            .attr("x", currentNodeX + distanceToFeature + m * cellWidth)
            .attr("y", currentNodeY + (dirCoefficient) * distanceToFeature / 2 - 12/2)
            .attr("width", cellWidth)
            .attr("height", 12)
            .attr("fill", featureColor(bias[m]))
            .attr("class", "bias-cell layer-inner-works")
            .attr("id", `fc-bias-cell-node-${nodeID}-dim-${m}`)
            .style("opacity", 1);
    }
    inner.append("rect").attr("x", currentNodeX + distanceToFeature).attr("y", currentNodeY + (dirCoefficient) * distanceToFeature / 2 - 12/2).attr("width", bias.length * cellWidth).attr("height", 12).attr("fill", "none").attr("class", "bias-frame layer-inner-works").attr("id", `fc-bias-frame-node-${nodeID}`).style("stroke-width", 1).style("stroke", "black").style("opacity", 1);
    // its addition
    const biasedAddition = addVector(multipliedFeature, bias);
    for(let l=0; l < biasedAddition.length; l++) inner.append("rect").attr("x", currentNodeX + distanceToFeature * 2 + multipliedFeature.length * cellWidth + l * cellWidth).attr("y", currentNodeY - 12/2).attr("width", cellWidth).attr("height", 12).attr("fill", featureColor(biasedAddition[l]))
    inner.append("rect").attr("x", currentNodeX + distanceToFeature * 2 + multipliedFeature.length * cellWidth).attr("y", currentNodeY - 12/2).attr("width", biasedAddition.length * cellWidth).attr("height", 12).attr("fill", "none").attr("class", "biased-addition-frame layer-inner-works").attr("id", `fc-biased-addition-frame-node-${nodeID}`).style("stroke-width", 1).style("stroke", "black").style("opacity", 1);
    // visualize activation function
    inner.append("line").attr("x1", currentNodeX + distanceToFeature * 2 + multipliedFeature.length * cellWidth + biasedAddition.length * cellWidth).attr("y1", currentNodeY).attr("x2", currentNodeX + distanceToFeature * 3 + multipliedFeature.length * cellWidth + biasedAddition.length * cellWidth).attr("y2", currentNodeY).attr("stroke", "black").attr("opacity", 1).attr("class", "activation-line layer-inner-works").lower();
}


