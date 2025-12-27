import * as d3 from 'd3';
import { extractFCNodeIndex, extractFeatureId, getMaxLayerID, transitFCLayer, transitFeatureLayers } from './pipeUtils';
import { visualizeInnerFCLayerSubpipe, visualizeInnerGNNLayerSubpipe } from './visualizationPipeline';
import { distanceToFeature } from '../utils/const';

var isExpandLayer = false;
var currentLayerID = -1;

const maxLayerNum = getMaxLayerID();
console.log("maxLayerNum:", maxLayerNum);


export function interactionPipeline(cellWidth: number, adjacencyMatrix: number[][], sortedGNNFeatures: any[], modelInfo: any) {
    // interaction pipes
    interactNodesAndLinksPipe(adjacencyMatrix);
    interactFCNodesAndLinksPipe(sortedGNNFeatures, adjacencyMatrix);
    interactLayerExpansionPipe(cellWidth, adjacencyMatrix, sortedGNNFeatures, modelInfo);
    interactFCExpansionPipe(cellWidth, sortedGNNFeatures);
}

export function interactNodesAndLinksPipe(adjacencyMatrix: number[][]) {
    const g = d3.select("#matvis");
    g.selectAll(".feature-layer")
        .on("mouseover", function(event: any, d: any) {
            if (isExpandLayer) return;
            const id = (this as HTMLElement).id;
            const match = extractFeatureId(id);

            if (!match) return;

            const ithLayer = Number(match[1]);
            const jthNode = Number(match[2]);

            const neighbors: number[] = adjacencyMatrix[jthNode];

            console.log("mouseover node this:", this, "layer:", ithLayer, "node:", jthNode);
            d3.select(this).select(".feature-layer-frame")
                .style("opacity", 1)
                .style("stroke", "black");
            
            for (let i=0; i<neighbors.length; i++){
                if (neighbors[i] === 0) continue;
                g.select(`#link-path-${ithLayer-1}-${i}-to-${jthNode}`)
                    .style("stroke", "black")
                    .style("opacity", 1);

                g.select(`#feature-layer-${ithLayer-1}-node-${i}`)
                    .select(".feature-layer-frame")
                    .style("opacity", 1);
            }

            if(ithLayer!=0)interactNodesAndActivateMatrixSubpipe(jthNode, adjacencyMatrix, "activate-multiple");
            else interactNodesAndActivateMatrixSubpipe(jthNode, adjacencyMatrix, "activate-single");
        })
        .on("mouseout", function(event: any, d: any) {
            if (isExpandLayer) return;
            console.log("mouseout node:", this);
            g.selectAll(".link-path")
                .style("stroke", "black")
                .style("opacity", 0.1);

            d3.selectAll(".feature-layer-frame")
                .style("opacity", 0.5)
                .style("stroke", "black");
            
            interactNodesAndDeactivateMatrixSubpipe();
        });
}

export function interactFCNodesAndLinksPipe(sortedGNNFeatures: any[], adjacencyMatrix: number[][]) {
    const g = d3.select("#matrix-svg");
    g.selectAll(".fc-feature-layer")
        .on("mouseover", function(event: any, d: any) {
            if (isExpandLayer) return;
            d3.select(this)
                .select(".fc-feature-layer-frame")
                .style("opacity", 1);
            
            const id = (this as HTMLElement).id;
            const match = id.match(/^fc-feature-layer-node-(\d+)$/);
            if (!match) return;
            g.select(`#link-path-fc-${match[1]}`).style("opacity", 1);
            g.select(`#feature-layer-${sortedGNNFeatures.length-1}-node-${match[1]}`).select(".feature-layer-frame").style("opacity", 1);
            interactNodesAndActivateMatrixSubpipe(Number(match[1]), adjacencyMatrix, "activate-single");
        })
        .on("mouseout", function(event: any, d: any) {
            if (isExpandLayer) return;
            d3.select(this)
                .select(".fc-feature-layer-frame")
                .style("opacity", 0.5);
            d3.selectAll(".link-path-fc")
                .style("opacity", 0.1);
            d3.selectAll(".feature-layer-frame")
                .style("opacity", 0.5);
            interactNodesAndDeactivateMatrixSubpipe();
        });
}

export function interactNodesAndActivateMatrixSubpipe(selectedNode: number, adjacencyMatrix: number[][], mode: string) {
    const g = d3.select("#matrix-svg");
    if (mode == "activate-single") g.select(`#adj-matrix-row-border-${selectedNode}`).style("opacity", 1);
    if (mode == "activate-multiple") {
        g.select(`#adj-matrix-col-border-${selectedNode}`).style("opacity", 1);
        for (let j = 0; j < adjacencyMatrix[selectedNode].length; j++) {
            if (adjacencyMatrix[selectedNode][j] === 1 && j !== selectedNode) {
                g.select(`#adj-matrix-row-border-${j}`).style("opacity", 1);
            }
        }
    }
}

export function interactNodesAndDeactivateMatrixSubpipe() {
    const g = d3.select("#matrix-svg");
    g.selectAll(".adj-matrix-row-border, .adj-matrix-col-border").style("opacity", 0);
}

export function interactLayerExpansionPipe(cellWidth: number, adjacencyMatrix: number[][], sortedGNNFeatures: any[][], modelInfo: any){
    const g = d3.select("#matvis");

    g.selectAll(".feature-layer")
        .on("click", function(event: any, d: any) {
            event.stopPropagation();
            isExpandLayer = !isExpandLayer;
            const id = (this as HTMLElement).id;
            const matchedID = extractFeatureId(id);
            const layerID = matchedID[1];
            const nodeID = matchedID[2];
            currentLayerID = Number(layerID);
            console.log("isExpandLayer click:", isExpandLayer, matchedID);
            d3.selectAll(".link-path, .link-path-fc").style("opacity", 0);
            d3.selectAll(".feature-layer, .fc-feature-layer")
                .style("opacity", 0.1)
                .attr("pointer-events", "none");
            d3.select(this).style("opacity", 1);
            for(let i = 0; i < adjacencyMatrix[nodeID].length; i++) {
                if (adjacencyMatrix[nodeID][i] === 1){
                    d3.select(`#feature-layer-${layerID-1}-node-${i}`).style("opacity", 1);
                }
            }
            const dist = 50 * 3 + sortedGNNFeatures[layerID-1][0].length * cellWidth + sortedGNNFeatures[layerID][0].length * cellWidth - 100;
            transitFeatureLayers(layerID, dist);
            let direction = "up";
            if (nodeID < (adjacencyMatrix.length)/2) direction = "down";
            visualizeInnerGNNLayerSubpipe(cellWidth, layerID, nodeID, adjacencyMatrix, sortedGNNFeatures, modelInfo, direction);
        });

    g.on("click", function(event: any, d: any) {
        if(isExpandLayer)isExpandLayer = false;
        console.log("isExpandLayer updated:", isExpandLayer);
        d3.selectAll(".link-path, .link-path-fc").style("opacity", 0.1);
        d3.selectAll(".feature-layer, .fc-feature-layer")
            .style("opacity", 1)
            .attr("pointer-events", "auto");
        d3.selectAll(".layer-inner-works-group").remove();
        if(currentLayerID !== -1 && currentLayerID != maxLayerNum+1)transitFeatureLayers(currentLayerID, 0);
        if(currentLayerID === maxLayerNum+1)transitFCLayer(0);
        currentLayerID = -1;
    });
}

export function interactFCExpansionPipe(cellWidth: number, sortedGNNFeatures: any[][]){
    const biasDim = 4;
    const g = d3.select("#matvis");
    g.selectAll(".fc-feature-layer")
        .on("click", function(event: any, d: any) {
            event.stopPropagation();
            isExpandLayer = true;
            currentLayerID = maxLayerNum + 1;
            const id = extractFCNodeIndex((this as HTMLElement).id);
            d3.selectAll(".link-path, .link-path-fc").style("opacity", 0);
            d3.selectAll(".feature-layer, .fc-feature-layer")
                .style("opacity", 0.1)
                .attr("pointer-events", "none");
            d3.select(this).style("opacity", 1);
            d3.select("#feature-layer-" + maxLayerNum + "-node-" + id).style("opacity", 1);
            // the problem for both transition distance is need to minus the 'gap=100' to align the view!!!
            const transitDistance = (sortedGNNFeatures[sortedGNNFeatures.length-1][0].length + biasDim * 2) * cellWidth + distanceToFeature * 3 - 100;
            transitFCLayer(transitDistance);
            let direction = "down";
            if (id < (sortedGNNFeatures[sortedGNNFeatures.length-1].length)/2) direction = "up";
            visualizeInnerFCLayerSubpipe(cellWidth, id, sortedGNNFeatures, direction);
        });
}


