import * as d3 from "d3";
import { injectSVG } from "./utils/pipeUtils";
import { computeFeatureLayerX, computeFeatureLayerY } from "./utils/geometryUtils";
import { distanceToFeature } from "../utils/const";
import { matrixTranspose, randomVector, scaleVector, vecMatMul, addVector, countOnes, divideVector } from "./utils/mathUtils";
import { curve, featureColor } from "./utils/const";
import { buildVisibleLayerNodeIds, extractSortedGNNLayerFeatures, getVisibleNodeIdsForLayer, SubgraphResult } from "./utils/dataProcessingUtils";

type LayerNodeIds = number[][];

export function visualizationPipeline(container: HTMLDivElement, cellWidth: number, cellHeight: number, adjacencyMatrix: number[][], intmData: any, linkList: any[], queries: number[][] = [], subgraphData: any, subgraphSample: any, mode: string) {
    // define parameters
    const gapSizeBetweenLayers = 100;
    console.log("subgraphData inside visualizationPipeline:", subgraphData);
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
    const visibleLayerNodeIds = buildVisibleLayerNodeIds(sortedGNNFeatures, subgraphData, subgraphSample);
    
    // visualization pipes
    visualizeMatrixPipe(container, adjacencyMatrix, visibleLayerNodeIds[0]);
    visualizeIntermediateFeaturePipe(container, cellWidth, cellHeight, gapSizeBetweenLayers, intmData, adjacencyMatrix, queries, subgraphData, subgraphSample, mode, visibleLayerNodeIds);
    visualizeLinksBetweenLayersPipe(container, linkList, gapSizeBetweenLayers, intmData, subgraphData, visibleLayerNodeIds);
    resizeSvgToContent(container);
}

export function visualizeMatrixPipe(container: HTMLDivElement, adjacencyMatrix: number[][], visibleNodeIds: number[]) {
    
    console.log("Adjacency Matrix in the Vis Pipe:", adjacencyMatrix);

    const g = d3.select(container);
    g.selectAll("*").remove();

    const width = 1;
    const height = 1;

    const svg = g
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                .attr("id", "matrix-svg");

    // visualize the matrix
    const startX = 50;
    const startY = 50;
    const cellSize = 20;

    for (let displayRow = 0; displayRow < visibleNodeIds.length; displayRow++) {
        const rowNodeId = visibleNodeIds[displayRow];
        for (let displayCol = 0; displayCol < visibleNodeIds.length; displayCol++) {
            const colNodeId = visibleNodeIds[displayCol];
            if (adjacencyMatrix[rowNodeId][colNodeId] === 1) {
                svg.append("rect")
                    .attr("x", startX + displayCol * cellSize)
                    .attr("y", startY + displayRow * cellSize)
                    .attr("width", cellSize)
                    .attr("height", cellSize)
                    .attr("fill", "rgb(105, 179, 162)")
                    .attr("class", "adj-matrix-cell")
                    .attr("id", `cell-${rowNodeId}-${colNodeId}`)
                    .style("stroke", "white")
                    .style("stroke-width", 1)
                    .style("opacity", 0.8);
            } else {
                svg.append("rect")
                    .attr("x", startX + displayCol * cellSize)
                    .attr("y", startY + displayRow * cellSize)
                    .attr("width", cellSize)
                    .attr("height", cellSize)
                    .attr("fill", "rgb(238, 238, 238)")
                    .attr("class", "adj-matrix-cell")
                    .attr("id", `cell-${rowNodeId}-${colNodeId}`)
                    .style("stroke", "white")
                    .style("stroke-width", 1)
                    .style("opacity", 0.8);
            }
        }
        svg.append("rect")
            .attr("x", startX)
            .attr("y", startY + displayRow * cellSize)
            .attr("width", cellSize * visibleNodeIds.length)
            .attr("height", cellSize)
            .attr("fill", "none")
            .attr("class", "adj-matrix-row-border")
            .attr("id", `adj-matrix-row-border-${rowNodeId}`)
            .style("stroke", "black")
            .style("stroke-width", 1)
            .style("opacity", 0);

        svg.append("rect")
            .attr("x", startX + displayRow * cellSize)
            .attr("y", startY)
            .attr("width", cellSize )
            .attr("height", cellSize * visibleNodeIds.length)
            .attr("fill", "none")
            .attr("class", "adj-matrix-col-border")
            .attr("id", `adj-matrix-col-border-${rowNodeId}`)
            .style("stroke", "black")
            .style("stroke-width", 1)
            .style("opacity", 0);
    }
    svg.selectAll(".adj-matrix-row-border, .adj-matrix-col-border").raise();
    
}

export function resizeSvgToContent(container: HTMLDivElement, padding: number = 24) {
    const svg = d3.select(container).select<SVGSVGElement>("svg");
    const svgNode = svg.node();
    if (!svgNode) {
        return;
    }

    const applyResize = () => {
        const bbox = svgNode.getBBox();
        const width = Math.max(1, Math.ceil(bbox.width + padding));
        const height = Math.max(1, Math.ceil(bbox.height + padding));
        const minX = Math.floor(bbox.x - padding / 2);
        const minY = Math.floor(bbox.y - padding / 2);

        svg
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", `${minX} ${minY} ${width} ${height}`);
    };

    requestAnimationFrame(applyResize);
    window.setTimeout(applyResize, 550);
}

export function visualizeIntermediateFeaturePipe(container: HTMLDivElement, cellWidth: number, cellHeight: number, gapXBetweenLayers: number, intmData: any, adjacencyMatrix: number[][], queries: number[][] = [], subgraphData: any, subgraphSample: any, mode: string, visibleLayerNodeIds: LayerNodeIds){
    console.log("inside visualizeIntermediateFeaturePipe", intmData, adjacencyMatrix);

    const svg = d3.select(container).select("svg");
    const startX = visibleLayerNodeIds[0].length * 20 + 20 + 50;
    const startY = 50;

    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
    console.log("Sorted GNN Layer Features:", sortedGNNFeatures);

    let layerX = startX;
    for(let i=0; i < sortedGNNFeatures.length; i++){
        const visibleNodeIds = visibleLayerNodeIds[i];
        console.log("visible nodes inside visualizeIntermediateFeaturePipe:", i, visibleNodeIds);
        if (i > 0)layerX += cellWidth * sortedGNNFeatures[i-1][0].length;
        const layerFeatures = sortedGNNFeatures[i];
        for (let displayIndex = 0; displayIndex < visibleNodeIds.length; displayIndex++) {
            const nodeId = visibleNodeIds[displayIndex];
            const layerY = startY + displayIndex * 20;
            const feature = layerFeatures[nodeId];
            if (!Array.isArray(feature)) continue;
            const g = svg.append("g").attr("class", "feature-layer").attr("id", `feature-layer-${i}-node-${nodeId}`);

            g.append("rect")
                .attr("x", layerX)
                .attr("y", layerY + (cellHeight/2))
                .attr("width", feature.length * cellWidth)
                .attr("height", cellHeight)
                .attr("fill", "none")
                .attr("class", "feature-layer-frame")
                .attr("id", `feature-layer-frame-${i}-node-${nodeId}`)
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
                    .attr("id", `feature-layer-${i}-node-${nodeId}-dim-${k}`)
                    .style("stroke-width", 0.5)
                    .style("stroke", "gray")
                    .style("stroke-opacity", 0.5)
                    .style("opacity", 1);
            }
            }
        layerX +=  (gapXBetweenLayers);
    }
    console.log("mode inside visualizeIntermediateFeaturePipe:", mode);
    if (mode == 'node') visualizeFCForNodeTaskSubpipe(container, layerX, intmData, subgraphData, subgraphSample);
    else if (mode == 'edge') visualizeFCForEdgeTaskSubpipe(container, layerX, intmData, queries);
    else if (mode == 'graph') visualizeFCForGraphTaskSubpipe(container, layerX, intmData);
}

export function visualizeLinksBetweenLayersPipe(
    container: HTMLDivElement,
    links: any,
    gapSize: number,
    intmData: any,
    subgraphData: SubgraphResult[],
    visibleLayerNodeIds: LayerNodeIds
){
    console.log("start visualizeLinksBetweenLayers");
    console.log("subgraphData inside visualizeLinksBetweenLayers:", subgraphData);
    // visualize links between GNN layers
    const svg = d3.select(container).select("svg");

    console.log("intmData inside visualizeLinksBetweenLayers:", intmData);
    console.log("sortedGNNFeatures:", extractSortedGNNLayerFeatures(intmData));


    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
    // const undirectLinks = removeRepeatLinks(links);

    const startX = 50 + visibleLayerNodeIds[0].length * 20 + 20;
    const startY = 50;

    let layerX = startX + sortedGNNFeatures[0][0].length * 6;

    // looping through layers
    for(let i=0; i < sortedGNNFeatures.length - 1; i++){
        // compute locations
        const prevLayerX = layerX;
        layerX += sortedGNNFeatures[i+1][0].length * 6 + gapSize;
        const midLayerX = (prevLayerX + layerX) / 2;
        const prevLayerVisibleNodes = visibleLayerNodeIds[i];
        const nextLayerVisibleNodes = visibleLayerNodeIds[i + 1];
        const prevLayerIndexMap = new Map(prevLayerVisibleNodes.map((nodeId, index) => [nodeId, index]));
        const nextLayerIndexMap = new Map(nextLayerVisibleNodes.map((nodeId, index) => [nodeId, index]));
        console.log("visible nodes inside visualizeLinksBetweenLayers:", i, nextLayerVisibleNodes);
        // looping through nodes in layer i
        for (let j = 0; j < links.length; j++) {
            const link = links[j];
            const sourceIdx = link.source;
            const targetIdx = link.target;

            if (prevLayerIndexMap.has(sourceIdx) && nextLayerIndexMap.has(targetIdx)) {
                const sourceY = startY + prevLayerIndexMap.get(sourceIdx)! * 20 + 12;
                const targetY = startY + nextLayerIndexMap.get(targetIdx)! * 20 + 12;

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
        }
        // visualize self-looping
        for (let displayIndex = 0; displayIndex < nextLayerVisibleNodes.length; displayIndex++) {
            const nodeId = nextLayerVisibleNodes[displayIndex];
            const layerY = startY + displayIndex * 20 + 12;
            svg.append("line")
                .attr("x1", layerX)
                .attr("y1", layerY)
                .attr("x2", layerX - 125)
                .attr("y2", layerY)
                .attr("stroke", "black")
                .attr("opacity", 0.1)
                .attr("fill", "none")
                .attr("class", "link-path")
                .attr("id", `link-path-${i}-${nodeId}-to-${nodeId}`)
                .lower();
        }
    }
}

export function visualizeFCForEdgeTaskSubpipe(container: HTMLDivElement, layerX: any, intmData: any, queries: number[][]){
    const sortedLayers = extractSortedGNNLayerFeatures(intmData);
    const lastLayerNum = sortedLayers[sortedLayers.length - 1].length;
    const fcLayerFeatures: any[][] = intmData[`decoder`];
    console.log("fc data", fcLayerFeatures, lastLayerNum);
    const layerY = 50;
    const prevLayerX = layerX - 100;
    const svg = d3.select(container).select("svg");
    console.log("queries for edge task:", queries, fcLayerFeatures);
    // visualize links
    for (let i=0; i < queries.length; i++){
        const nodeA = queries[i][0];
        const nodeB = queries[i][1];
        const layerYA = layerY + nodeA * 20 + 12;
        const layerYB = layerY + nodeB * 20 + 12;
        const layerYMid = (layerYA + layerYB) / 2;
        svg.append("path")
            .attr("d", curve([
                [prevLayerX, layerYA],
                [prevLayerX + 50, layerYA],
                [prevLayerX + 50, layerYMid],
                [layerX, layerYMid],
            ])).attr("stroke", "black").attr("opacity", 0.1).attr("fill", "none").attr("class", `link-path-fc link-path-fc-${i}`).lower();
        svg.append("path")
            .attr("d", curve([
                [prevLayerX, layerYB],
                [prevLayerX + 50, layerYB],
                [prevLayerX + 50, layerYMid],
                [layerX, layerYMid],
            ])).attr("stroke", "black").attr("opacity", 0.1).attr("fill", "none").attr("class", `link-path-fc link-path-fc-${i}`).lower();
        // visualize final probabilities output
        const prob = Math.random();
        const probArr = [1 - prob, prob];
        const g = svg.append("g").attr("class", "fc-feature-layer").attr("id", `fc-feature-layer-node-${i}`);
        for(let j=0; j < probArr.length; j++){
            g.append("rect")
                .attr("x", layerX + j * 6)
                .attr("y", layerYMid - 6)
                .attr("width", 6)
                .attr("height", 12)
                .attr("fill", featureColor(probArr[j]))
                .attr("class", "fc-feature-cell")
                .attr("id", `fc-feature-layer-node-${i}-dim-${j}`)
                .style("stroke-width", 0.5)
                .style("stroke", "gray")
                .style("stroke-opacity", 0.5)
                .style("opacity", 1);
        }
        g.append("rect")
            .attr("x", layerX)
            .attr("y", layerYMid - 6)
            .attr("width", probArr.length * 6)
            .attr("height", 12)
            .attr("fill", "none")
            .attr("class", "fc-feature-layer-frame")
            .attr("id", `fc-feature-layer-frame-node-${i}`)
            .style("stroke-width", 1)
            .style("stroke", "black")
            .style("opacity", 0.5);
    }

}

export function visualizeFCForGraphTaskSubpipe(container: HTMLDivElement, layerX: any, intmData: any){
    const sortedLayers = extractSortedGNNLayerFeatures(intmData);
    const lastLayerNum = sortedLayers[sortedLayers.length - 1].length;
    const fcLayerFeatures: any[][] = intmData[`conv4`];
    console.log("fc data", fcLayerFeatures, lastLayerNum);
    const prevLayerX = layerX - 100;
    const layerY = 50;
    const svg = d3.select(container).select("svg");
    const midLayerY = layerY + (fcLayerFeatures.length * 20) / 2;
    for (let i=0; i < fcLayerFeatures.length; i++){
        const curLayerY = layerY + i * 20 + 12;
        svg.append("path")
            .attr("d", curve([
                [prevLayerX, curLayerY],
                [prevLayerX + 50, curLayerY],
                [prevLayerX + 50, midLayerY],
                [layerX, midLayerY],
            ])).attr("stroke", "black").attr("opacity", 0.1).attr("fill", "none").attr("class", "agg-link-path-fc").attr("id", `agg-link-path-fc-${i}`).lower();
    }
    let vec = Array(fcLayerFeatures[0].length).fill(0);
    for (let i=0; i < fcLayerFeatures.length; i++)
        vec = addVector(vec, fcLayerFeatures[i]);
    vec = divideVector(vec, fcLayerFeatures.length);
    console.log("averaged graph feature vector:", vec);
    const g = svg.append("g").attr("class", "agg-feature-layer").attr("id", `agg-feature-layer-node-graph`);
    for(let j=0; j < vec.length; j++){
        g.append("rect")
            .attr("x", layerX + j * 6)
            .attr("y", midLayerY - 6)
            .attr("width", 6)
            .attr("height", 12)
            .attr("fill", featureColor(vec[j]))
            .attr("class", "agg-feature-cell")
            .attr("id", `agg-feature-layer-node-graph-dim-${j}`)
            .style("stroke-width", 0.5)
            .style("stroke", "gray")
            .style("stroke-opacity", 0.5)
            .style("opacity", 1);
    }
    g.append("rect")
        .attr("x", layerX)
        .attr("y", midLayerY - 6)
        .attr("width", vec.length * 6)
        .attr("height", 12)
        .attr("fill", "none")
        .attr("class", "agg-feature-layer-frame")
        .attr("id", `agg-feature-layer-frame-node-graph`)
        .style("stroke-width", 1)
        .style("stroke", "black")
        .style("opacity", 0.5);

    const resultVec = randomVector(4);
    visualizeSingleFCSubpipe(layerX + 100, midLayerY - 12, resultVec, 0, svg);
}

export function visualizeFCForNodeTaskSubpipe(container: HTMLDivElement, layerX: number, intmData: any, subgraphData: SubgraphResult[], subgraphSample: any){
    console.log("inside visualizeFCFeaturesPipe", intmData);
    // get the last layer number from intmData
    const sortedLayers = extractSortedGNNLayerFeatures(intmData);
    const lastLayerNum = sortedLayers[sortedLayers.length - 1].length;
    const fcLayerFeatures: any[][] = intmData[`softmax`]; // TODO: make it more general 
    console.log("fc data", fcLayerFeatures, lastLayerNum);
    const layerY = 50;
    const svg = d3.select(container).select("svg");
    const visibleNodeIds = getVisibleNodeIdsForLayer(fcLayerFeatures, subgraphData, subgraphData.length - 1, subgraphSample);
    for (let displayIndex = 0; displayIndex < visibleNodeIds.length; displayIndex++) {
        const nodeId = visibleNodeIds[displayIndex];
        const feature: any[] | undefined = fcLayerFeatures[nodeId];
        console.log("fc feature:", feature, nodeId);
        if (!Array.isArray(feature)) continue;
        visualizeSingleFCSubpipe(layerX, layerY, feature, displayIndex, svg, nodeId);
    }
}

function visualizeSingleFCSubpipe(layerX: number, layerY: number, feature: any[], displayIndex: number, svg: any, nodeId: number = displayIndex) {
    const g = svg.append("g").attr("class", "fc-feature-layer").attr("id", `fc-feature-layer-node-${nodeId}`);

    g.append("rect")
        .attr("x", layerX)
        .attr("y", layerY + displayIndex * 20 + 6)
        .attr("width", feature.length * 6)
        .attr("height", 12)
        .attr("fill", "none")
        .attr("class", "fc-feature-layer-frame")
        .attr("id", `fc-feature-layer-frame-node-${nodeId}`)
        .style("stroke-width", 2)
        .style("stroke", "black")
        .style("opacity", 0.5);

    for (let j = 0; j < feature.length; j++) {
        g.append("rect")
            .attr("x", layerX + j * 6)
            .attr("y", layerY + displayIndex * 20 + 6)
            .attr("width", 6)
            .attr("height", 12)
            .attr("fill", featureColor(feature[j]))
            .attr("class", "fc-feature-cell")
            .attr("id", `fc-feature-layer-node-${nodeId}-dim-${j}`)
            .style("stroke-width", 0.5)
            .style("stroke", "gray")
            .style("stroke-opacity", 0.5)
            .style("opacity", 1);
    }

    svg.append("line")
        .attr("x1", layerX)
        .attr("y1", layerY + displayIndex * 20 + 12)
        .attr("x2", layerX - 100)
        .attr("y2", layerY + displayIndex * 20 + 12)
        .attr("stroke", "black")
        .attr("opacity", 0.1)
        .attr("fill", "none")
        .attr("class", "link-path-fc")
        .attr("id", `link-path-fc-${nodeId}`)
        .lower();
}

export function visualizeInnerGNNLayerSubpipe(container: HTMLDivElement, cellWidth: number, layerID: number, nodeID: number, adjacencyMatrix: number[][], sortedGNNFeatures: any[][], modelInfo: any, direction: string, layerTranslateX: number = 0, visibleLayerNodeIds: LayerNodeIds = []){
    console.log("inside layer modelInfo:", modelInfo, layerID);
    const distanceBetweenFeatures = 50;
    const g = d3.select(container).select("svg");
    const inner = g.append("g").attr("class", "layer-inner-works-group").attr("id", `layer-inner-works-group-layer-${layerID}-node-${nodeID}`);
    const currentLayerVisibleNodes = visibleLayerNodeIds[layerID] ?? sortedGNNFeatures[layerID].map((_, index) => index);
    const previousLayerVisibleNodes = visibleLayerNodeIds[layerID - 1] ?? sortedGNNFeatures[layerID - 1].map((_, index) => index);
    const currentDisplayIndex = currentLayerVisibleNodes.indexOf(nodeID);

    if (currentDisplayIndex === -1) {
        inner.remove();
        return;
    }

    const startX = currentLayerVisibleNodes.length > 0 ? visibleLayerNodeIds[0].length * 20 + 20 + 50 : 50;
    let currentNodeX = startX;
    for (let i = 0; i < layerID; i++) {
        currentNodeX += sortedGNNFeatures[i][0].length * cellWidth + 100;
    }
    currentNodeX += layerTranslateX;
    const currentNodeY = 50 + currentDisplayIndex * 20 + 12;

    let locations = [];

    let aggregatedFeature: number[] = Array(sortedGNNFeatures[layerID-1][0].length).fill(0);
    let degreeMultipliers: number[] = [];

    let dirCoefficient = 1;
    if (direction === "up") dirCoefficient = -1;

    let previousLayerX = startX;
    for (let i = 0; i < layerID - 1; i++) {
        previousLayerX += sortedGNNFeatures[i][0].length * cellWidth + 100;
    }
    previousLayerX += sortedGNNFeatures[layerID - 1][0].length * cellWidth;

    for(let j = 0; j < adjacencyMatrix[nodeID].length; j++) {
        if (adjacencyMatrix[nodeID][j] === 1){
            const targetDisplayIndex = previousLayerVisibleNodes.indexOf(j);
            if (targetDisplayIndex === -1) continue;
            const targetNodeX = previousLayerX;
            const targetNodeY = 50 + targetDisplayIndex * 20 + 12;
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
        inner.append("path")
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
    const weightMatrix:number[][] = matrixTranspose(modelInfo[`conv${layerID}`]["weight"]);
    console.log("weightMatrix:", weightMatrix);
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
    const bias = modelInfo[`conv${layerID}`]["bias"];;
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

export function visualizeInnerFCLayerSubpipe(container: HTMLDivElement, cellWidth: number, nodeID: number, sortedGNNFeatures: any[][], modelInfo: any, direction: string, mode: string){
    console.log("inside fc visualizeInnerFCLayerSubpipe", modelInfo);
    let startX = sortedGNNFeatures[0].length * 20 + 20 + 50;

    const weightMatrix:number[][] = matrixTranspose(modelInfo['classifier']["weight"]);
    const bias = modelInfo['classifier']["bias"];
    let currentNodeX = computeFeatureLayerX(startX, sortedGNNFeatures.length, cellWidth, 100, sortedGNNFeatures);
    let currentNodeY = computeFeatureLayerY(nodeID, 50, 20);

    const inner = d3.select(container).select("svg").append("g").attr("class", "layer-inner-works-group");

    let dirCoefficient = 1;
    if (direction === "up") dirCoefficient = -1;

    if (mode == 'graph'){
        currentNodeX = currentNodeX + 100;
        currentNodeY = (computeFeatureLayerY(0, 50, 20) + computeFeatureLayerY(sortedGNNFeatures[sortedGNNFeatures.length-2].length - 1, 50, 20)) / 2;
    }  

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
