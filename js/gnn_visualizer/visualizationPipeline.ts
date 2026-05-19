import * as d3 from "d3";
import { injectSVG } from "./utils/pipeUtils";
import { computeFeatureLayerX, computeFeatureLayerY } from "./utils/geometryUtils";
import { distanceToFeature } from "../utils/const";
import { matrixTranspose, vecMatMul, addVector } from "./utils/mathUtils";
import { curve, featureColor } from "./utils/const";
import { aggregateNeighborFeatures } from "./utils/aggregationUtils";
import {
    normalizeLayerBias,
    normalizeLayerWeightMatrix,
} from "./utils/layerRenderingUtils";
import {
    extractSortedGNNLayerFeatures,
    getEdgeOutputFeature,
    getGraphAggregationInfo,
    getGraphOutputFeature,
    processSubgraphSequenceDataPipe,
    SubgraphResult,
} from "./utils/dataProcessingUtils";

const weightMatrixCellStroke = "#d8dedb";
const weightMatrixCellStrokeWidth = 0.35;
const sampledOutIconScale = 1.2;
const sampledOutIconGap = 2;

function isGraphSAGELayer(layerInfo: any) {
    const type = String(layerInfo?.type ?? "").toLowerCase();
    return type === "sageconv" || type === "graphsageconv";
}

function sampledOutNodeSet(layerInfo: any) {
    const candidates = [
        layerInfo?.sampled_out_nodes,
        layerInfo?.sampledOutNodes,
        layerInfo?.sample_out_nodes,
        layerInfo?.sampleOutNodes,
        layerInfo?.sampling?.sampled_out_nodes,
        layerInfo?.sampling?.sampledOutNodes,
    ];

    for (const candidate of candidates) {
        if (!Array.isArray(candidate)) {
            continue;
        }

        return new Set(
            candidate
                .map((value) => Number(value))
                .filter((value) => Number.isInteger(value))
        );
    }

    return new Set<number>();
}

export function visualizationPipeline(container: HTMLDivElement, cellWidth: number, cellHeight: number, adjacencyMatrix: number[][], intmData: any, linkList: any[], queries: number[][] = [], subgraphData: any, subgraphSample: any, mode: string, nodeLabels: string[] = [], messagePassingDepth: number = 4) {
    // define parameters
    const gapSizeBetweenLayers = 100;

    subgraphData = subgraphData ?? processSubgraphSequenceDataPipe(adjacencyMatrix, queries, messagePassingDepth);
    console.log("subgraphData inside visualizationPipeline:", subgraphData);
    
    // visualization pipes
    visualizeMatrixPipe(container, adjacencyMatrix, nodeLabels);
    visualizeIntermediateFeaturePipe(container, cellWidth, cellHeight, gapSizeBetweenLayers, intmData, adjacencyMatrix, queries, subgraphData, subgraphSample, mode);
    visualizeLinksBetweenLayersPipe(container, linkList, gapSizeBetweenLayers, intmData, subgraphData, subgraphSample, cellWidth);
    resizeSvgToContent(container);
}

function appendMatrixLabels(svg: any, startX: number, startY: number, cellSize: number, labels: string[]) {
    labels.forEach((label, index) => {
        const y = startY + index * cellSize + cellSize / 2 + 3;
        const x = startX + index * cellSize + cellSize / 2;

        svg.append("text")
            .attr("x", startX - 14)
            .attr("y", y)
            .attr("text-anchor", "end")
            .attr("font-size", 8)
            .attr("fill", "#203d35")
            .text(label);

        svg.append("text")
            .attr("x", startX - 4)
            .attr("y", y)
            .attr("text-anchor", "end")
            .attr("font-size", 6)
            .attr("fill", "#6b817b")
            .text(index);

        svg.append("text")
            .attr("x", x)
            .attr("y", startY - 6)
            .attr("text-anchor", "start")
            .attr("font-size", 7)
            .attr("fill", "#203d35")
            .attr("transform", `rotate(-90 ${x} ${startY - 6})`)
            .text(label);
    });
}

function appendFeatureLegend(svg: any, x: number, y: number) {
    const width = 120;
    const height = 7;
    const steps = 40;

    for (let index = 0; index < steps; index++) {
        const value = -1.5 + (3 * index) / (steps - 1);
        svg.append("rect")
            .attr("x", x + (index * width) / steps)
            .attr("y", y)
            .attr("width", width / steps + 0.5)
            .attr("height", height)
            .attr("fill", featureColor(value));
    }

    [[-1.5, "-1.50"], [0, "0.00"], [1.5, "1.50"]].forEach(([value, label]) => {
        const tickX = x + ((Number(value) + 1.5) / 3) * width;
        svg.append("text")
            .attr("x", tickX)
            .attr("y", y + 20)
            .attr("text-anchor", "middle")
            .attr("font-size", 7)
            .attr("fill", "#6b817b")
            .text(label);
    });
}

export function visualizeMatrixPipe(container: HTMLDivElement, adjacencyMatrix: number[][], nodeLabels: string[] = []) {
    
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
    const labels = adjacencyMatrix.map((_, index) => nodeLabels[index] ?? String(index));

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
    appendMatrixLabels(svg, startX, startY, cellSize, labels);
    appendFeatureLegend(svg, startX + adjacencyMatrix.length * cellSize + 60, 20);
    
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

export function visualizeIntermediateFeaturePipe(container: HTMLDivElement, cellWidth: number, cellHeight: number, gapXBetweenLayers: number, intmData: any, adjacencyMatrix: number[][], queries: number[][] = [], subgraphData: any, subgraphSample: any, mode: string){
    console.log("inside visualizeIntermediateFeaturePipe", intmData, adjacencyMatrix);

    const svg = d3.select(container).select("svg");
    const startX = adjacencyMatrix.length * 20 + 20 + 50;
    const startY = 50;

    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
    console.log("Sorted GNN Layer Features:", sortedGNNFeatures);

    let layerX = startX;
    for(let i=0; i < sortedGNNFeatures.length; i++){
        const subgraph = subgraphData[i];
        console.log("subgraph inside visualizeIntermediateFeaturePipe:", i, subgraph.nodes);
        const layerFeatures = sortedGNNFeatures[i];
        for(let j=0; j < layerFeatures.length; j++){
            if (!subgraphSample || subgraph.nodes.includes(j)) {
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
        }
        layerX += (layerFeatures[0]?.length ?? 0) * cellWidth + gapXBetweenLayers;
    }
    console.log("mode inside visualizeIntermediateFeaturePipe:", mode);
    if (mode == 'node') visualizeFCForNodeTaskSubpipe(container, layerX, intmData);
    else if (mode == 'edge') visualizeFCForEdgeTaskSubpipe(container, layerX, intmData, queries);
    else if (mode == 'graph') visualizeFCForGraphTaskSubpipe(container, layerX, intmData);
}

export function visualizeLinksBetweenLayersPipe(
    container: HTMLDivElement,
    links: any,
    gapSize: number,
    intmData: any,
    subgraphData: SubgraphResult[],
    subgraphSample: any,
    cellWidth: number
){
    console.log("start visualizeLinksBetweenLayers");
    console.log("subgraphData inside visualizeLinksBetweenLayers:", subgraphData);
    // visualize links between GNN layers
    const svg = d3.select(container).select("svg");

    console.log("intmData inside visualizeLinksBetweenLayers:", intmData);
    console.log("sortedGNNFeatures:", extractSortedGNNLayerFeatures(intmData));


    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);
    // const undirectLinks = removeRepeatLinks(links);

    const getLayerLeftX = (layerIndex: number) => {
        let layerX = 50 + sortedGNNFeatures[0].length * 20 + 20;
        for (let index = 0; index < layerIndex; index++) {
            layerX += (sortedGNNFeatures[index][0]?.length ?? 0) * cellWidth + gapSize;
        }
        return layerX;
    };
    const getLayerRightX = (layerIndex: number) =>
        getLayerLeftX(layerIndex) + (sortedGNNFeatures[layerIndex][0]?.length ?? 0) * cellWidth;
    const startY = 50;

    // looping through layers
    for(let i=0; i < sortedGNNFeatures.length - 1; i++){
        // compute locations
        const sourceRightX = getLayerRightX(i);
        const targetLeftX = getLayerLeftX(i + 1);
        const midLayerX = (sourceRightX + targetLeftX) / 2;
        const subgraph = subgraphData[i+1];
        console.log("subgraph inside visualizeLinksBetweenLayers:", i, subgraph.nodes);
        // looping through nodes in layer i
        for (let j = 0; j < links.length; j++) {
            const link = links[j];
            const sourceIdx = link.source;
            const targetIdx = link.target;

            console.log("subgraphSample:", subgraphSample, subgraph.nodes.includes(targetIdx));

            if (!subgraphSample || subgraph.nodes.includes(targetIdx)) {
                const sourceY = startY + sourceIdx * 20 + 12;
                const targetY = startY + targetIdx * 20 + 12;

                const pathStart: [number, number] = [sourceRightX, sourceY];
                const pathEnd: [number, number] = [targetLeftX, targetY];

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
        for(let n=0; n < sortedGNNFeatures[i].length; n++){
            if (!subgraphSample || subgraph.nodes.includes(n)) {
                const layerY = startY + n * 20 + 12;
                svg.append("line")
                    .attr("x1", sourceRightX)
                    .attr("y1", layerY)
                    .attr("x2", targetLeftX)
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
        const probArr = getEdgeOutputFeature(intmData, sortedLayers, queries, i);
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
    const fcLayerFeatures = sortedLayers[sortedLayers.length - 1] ?? [];
    const graphAggregation = getGraphAggregationInfo(intmData, fcLayerFeatures);
    if (!graphAggregation) {
        return;
    }
    console.log("graph aggregation data", graphAggregation, fcLayerFeatures.length);
    const prevLayerX = layerX - 100;
    const layerY = 50;
    const svg = d3.select(container).select("svg");
    const midLayerY = layerY + (Math.max(1, fcLayerFeatures.length) * 20) / 2;
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
    const vec = graphAggregation.feature;
    console.log("graph aggregation feature vector:", vec);
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
    g.append("text")
        .attr("x", layerX)
        .attr("y", midLayerY + 34)
        .text(graphAggregation.label)
        .attr("class", "graph-aggregation-label")
        .attr("id", "graph-aggregation-label")
        .style("font-size", "14px")
        .style("font-weight", 700)
        .style("fill", "#7f7f7f");

    const resultVec = getGraphOutputFeature(intmData);
    if (resultVec) {
        visualizeSingleFCSubpipe(layerX + 100, midLayerY - 12, resultVec, 0, svg);
    }
}

export function visualizeFCForNodeTaskSubpipe(container: HTMLDivElement, layerX: number, intmData: any){
    console.log("inside visualizeFCFeaturesPipe", intmData);
    // get the last layer number from intmData
    const sortedLayers = extractSortedGNNLayerFeatures(intmData);
    const lastLayerNum = sortedLayers[sortedLayers.length - 1].length;
    const fcLayerFeatures: any[][] = intmData[`softmax`]; // TODO: make it more general 
    console.log("fc data", fcLayerFeatures, lastLayerNum);
    const layerY = 50;
    const svg = d3.select(container).select("svg");
    for(let i=0; i < fcLayerFeatures.length; i++){
        const feature: any[] | undefined = fcLayerFeatures[i];
        console.log("fc feature:", feature, i);
        if (!Array.isArray(feature)) continue;
        visualizeSingleFCSubpipe(layerX, layerY, feature, i, svg);
    }
}

function visualizeSingleFCSubpipe(layerX: number, layerY: number, feature: any[], i: number, svg: any) {
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

    for (let j = 0; j < feature.length; j++) {
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

export function visualizeInnerGNNLayerSubpipe(container: HTMLDivElement, cellWidth: number, layerID: number, nodeID: number, adjacencyMatrix: number[][], sortedGNNFeatures: any[][], modelInfo: any, direction: string){
    console.log("inside layer modelInfo:", modelInfo, layerID);
    const distanceBetweenFeatures = 50;
    const gapXBetweenLayers = 100;
    const startX = adjacencyMatrix.length * 20 + 20 + 50;
    const g = d3.select(container).select("svg");
    const inner = g.append("g").attr("class", "layer-inner-works-group").attr("id", `layer-inner-works-group-layer-${layerID}-node-${nodeID}`);

    const currentNodeX = computeFeatureLayerX(startX, layerID, cellWidth, gapXBetweenLayers, sortedGNNFeatures) - 2;
    const currentNodeY = computeFeatureLayerY(nodeID, 50, 20);

    let dirCoefficient = 1;
    if (direction === "up") dirCoefficient = -1;

    const layerInfo = modelInfo?.[`conv${layerID}`];
    const sampledOutNodes = isGraphSAGELayer(layerInfo) ? sampledOutNodeSet(layerInfo) : new Set<number>();
    const aggregationResult = aggregateNeighborFeatures(
        adjacencyMatrix,
        sortedGNNFeatures,
        layerID,
        nodeID,
        layerInfo
    );
    const aggregatedFeature = aggregationResult.aggregatedFeature;
    console.log("aggregatedFeature", aggregatedFeature);
    const firstIntersect: [number, number] = [currentNodeX + distanceBetweenFeatures, currentNodeY];
    const sourceNodeX = currentNodeX;
    // visualize aggregated links
    const controlX = sourceNodeX + (firstIntersect[0] - sourceNodeX) / 2;
    const ctrlPointForCurrentNode: [number, number] = [controlX, currentNodeY];
    for(let k = 0; k < aggregationResult.contributions.length; k++){
        const contribution = aggregationResult.contributions[k];
        const isSampledOut = sampledOutNodes.has(contribution.nodeIndex);
        const sourceNodeY = computeFeatureLayerY(contribution.nodeIndex, 50, 20);
        const ctrlPointForSourceNode: [number, number] = [controlX, sourceNodeY];
        const aggregationPath = inner.append("path")
            .attr("d", curve([[sourceNodeX, sourceNodeY], ctrlPointForSourceNode, ctrlPointForCurrentNode, firstIntersect]))
            .attr("stroke", isSampledOut ? "gray" : "black")
            .attr("opacity", 1)
            .attr("fill", "none")
            .attr("class", `link-path-aggregated layer-inner-works${isSampledOut ? " sampled-out-link" : ""}`)
            .attr("id", `link-path-aggregated-${layerID}-${nodeID}-to-${k}`)
            .lower();
        if (isSampledOut) {
            aggregationPath.attr("stroke-dasharray", "3,2");
        }
        const multiplierText = inner.append("text")
            .attr("x", sourceNodeX + 3)
            .attr("y", sourceNodeY - 6)
            .text(contribution.label)
            .attr(
                "class",
                `degree-multiplier-text layer-inner-works${aggregationResult.kind === "attention" ? " attention-coefficient-text" : ""}`
            )
            .attr("id", `degree-multiplier-text-${layerID}-${nodeID}-to-${k}`)
            .style("font-size", "6px");
        if (aggregationResult.kind === "attention") {
            multiplierText
                .style("fill", "#6e09cd")
                .style("font-weight", 700);
        }
        multiplierText.lower();
        if (isSampledOut) {
            const sourceFeatureWidth =
                (sortedGNNFeatures[layerID - 1]?.[contribution.nodeIndex]?.length ??
                    sortedGNNFeatures[layerID - 1]?.[0]?.length ??
                    0) * cellWidth;
            const sourceNodeLeftX = sourceNodeX - sourceFeatureWidth;
            const iconHalfWidth = (10 * sampledOutIconScale) / 2;
            const icon = injectSVG(
                inner,
                sourceNodeLeftX - sampledOutIconGap - iconHalfWidth,
                sourceNodeY,
                "./assets/sampling.svg",
                "sampling-icon sampled-out-node-icon layer-inner-works",
                sampledOutIconScale
            );
            if (icon instanceof Promise) {
                icon.then((node) => d3.select(node).append("title").text("Sampled out"));
            }
        }
    }
    // visualize aggregated feature
    const aggregatedFeatureGroup = inner.append("g").attr("class", "aggregated-feature-layer layer-inner-works").attr("id", `aggregated-feature-layer-layer-${layerID}-node-${nodeID}`);
    aggregatedFeatureGroup.append("text")
        .attr("x", currentNodeX + distanceBetweenFeatures)
        .attr("y", currentNodeY - 10)
        .text(`agg: ${aggregationResult.label}`)
        .attr("class", "aggregation-kind-text layer-inner-works")
        .attr("id", `aggregation-kind-text-${layerID}-${nodeID}`)
        .style("font-size", "7px")
        .style("fill", "#203d35");
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
    const weightMatrix = normalizeLayerWeightMatrix(layerInfo, aggregatedFeature.length);
    if (!weightMatrix) {
        return;
    }
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
                .style("stroke", weightMatrixCellStroke)
                .style("stroke-width", weightMatrixCellStrokeWidth)
                .style("opacity", 1);
        }
    }
    // visualize bias and actiivation function
    const multipliedFeature = vecMatMul(aggregatedFeature, weightMatrix);
    const bias = normalizeLayerBias(layerInfo, multipliedFeature.length);
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
    injectSVG(
        inner,
        currentNodeX + distanceBetweenFeatures*2.5 + aggregatedFeature.length * cellWidth + multipliedFeature.length * cellWidth,
        currentNodeY,
        "./assets/relu.svg",
        "activation-icon layer-inner-works"
    );
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
    injectSVG(
        inner,
        currentNodeX + distanceToFeature / 2,
        currentNodeY,
        "./assets/matmul.svg",
        "matmul-icon layer-inner-works"
    );
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
                .style("stroke", weightMatrixCellStroke)
                .style("stroke-width", weightMatrixCellStrokeWidth)
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
    injectSVG(
        inner,
        currentNodeX + distanceToFeature * 2.5 + multipliedFeature.length * cellWidth + biasedAddition.length * cellWidth,
        currentNodeY,
        "./assets/relu.svg",
        "activation-icon layer-inner-works"
    );
}
