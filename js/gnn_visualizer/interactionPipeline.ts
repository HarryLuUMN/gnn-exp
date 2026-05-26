import * as d3 from 'd3';
import { transitFCLayer, transitFeatureLayers } from './utils/animationUtils';
import { extractFCNodeIndex, extractFeatureId } from './utils/dataProcessingUtils';
import { resizeSvgToContent, visualizeInnerFCLayerSubpipe, visualizeInnerGNNLayerSubpipe } from './visualizationPipeline';
import { distanceToFeature } from '../utils/const';

type InteractionState = {
    currentLayerID: number;
    isExpandLayer: boolean;
};

export function interactionPipeline(container: HTMLDivElement, cellWidth: number, adjacencyMatrix: number[][], sortedGNNFeatures: any[], modelInfo: any, mode: string, queries: number[][] = [[12, 18]]) {
    const state: InteractionState = {
        currentLayerID: -1,
        isExpandLayer: false,
    };
    // interaction pipes
    interactNodesAndLinksPipe(container, adjacencyMatrix, state);
    interactFCNodesAndLinksPipe(container, sortedGNNFeatures, adjacencyMatrix, mode, queries, state);
    interactLayerExpansionPipe(container, cellWidth, adjacencyMatrix, sortedGNNFeatures, modelInfo, state);
    interactFCExpansionPipe(container, cellWidth, sortedGNNFeatures, modelInfo, mode, state);
}

export function interactNodesAndLinksPipe(container: HTMLDivElement, adjacencyMatrix: number[][], state: InteractionState) {
    const g = d3.select(container);
    g.selectAll(".feature-layer")
        .on("mouseover", function() {
            if (state.isExpandLayer) return;
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

            if(ithLayer!=0)interactNodesAndActivateMatrixSubpipe(container, jthNode, adjacencyMatrix, "activate-multiple");
            else interactNodesAndActivateMatrixSubpipe(container, jthNode, adjacencyMatrix, "activate-single");
        })
        .on("mouseout", function() {
            if (state.isExpandLayer) return;
            console.log("mouseout node:", this);
            g.selectAll(".link-path")
                .style("stroke", "black")
                .style("opacity", 0.1);

            g.selectAll(".feature-layer-frame")
                .style("opacity", 0.5)
                .style("stroke", "black");
            
            interactNodesAndDeactivateMatrixSubpipe(container);
        });
}

export function interactFCNodesAndLinksPipe(container: HTMLDivElement, sortedGNNFeatures: any[], adjacencyMatrix: number[][], mode: string, queries: number[][] = [[12, 18]], state: InteractionState) {
    const g = d3.select(container).select("svg");
    g.selectAll(".fc-feature-layer")
        .on("mouseover", function() {
            if (state.isExpandLayer) return;
            d3.select(this)
                .select(".fc-feature-layer-frame")
                .style("opacity", 1);
            
            const id = (this as HTMLElement).id;
            const match = id.match(/^fc-feature-layer-node-(\d+)$/);
            if (!match) return;
            if (mode == "node") {
                g.select(`#link-path-fc-${match[1]}`).style("opacity", 1);
                g.select(`#feature-layer-${sortedGNNFeatures.length-1}-node-${match[1]}`).select(".feature-layer-frame").style("opacity", 1);
                interactNodesAndActivateMatrixSubpipe(container, Number(match[1]), adjacencyMatrix, "activate-single");
            } else if (mode == "edge") {
                g.selectAll(`.link-path-fc-${match[1]}`).style("opacity", 1);
                g.select(`#feature-layer-${sortedGNNFeatures.length-1}-node-${match[1]}`).select(".feature-layer-frame").style("opacity", 1);
                console.log("match[1]:", queries[Number(match[1])]);
                console.log("queries:", queries[Number(match[1])][0], queries[Number(match[1])][1]);
                interactNodesAndActivateMatrixSubpipe(container, queries[Number(match[1])][0], adjacencyMatrix, "activate-single");
                interactNodesAndActivateMatrixSubpipe(container, queries[Number(match[1])][1], adjacencyMatrix, "activate-single");
                g.select(`#feature-layer-frame-${sortedGNNFeatures.length-1}-node-${queries[Number(match[1])][0]}`).style("opacity", 1);
                g.select(`#feature-layer-frame-${sortedGNNFeatures.length-1}-node-${queries[Number(match[1])][1]}`).style("opacity", 1);
            } else if (mode == "graph") {
                g.select(".agg-feature-layer-frame").style("opacity", 1);
                g.select("#fc-feature-layer-frame-node-0").style("opacity", 1);
                g.select("#link-path-fc-0").style("opacity", 1);
            }
        })
        .on("mouseout", function() {
            if (state.isExpandLayer) return;
            d3.select(this)
                .select(".fc-feature-layer-frame")
                .style("opacity", 0.5);
            g.selectAll(".link-path-fc")
                .style("opacity", 0.1);
            g.selectAll(".feature-layer-frame")
                .style("opacity", 0.5);
            g.select(".agg-feature-layer-frame").style("opacity", 0.5);
            interactNodesAndDeactivateMatrixSubpipe(container);
        });

    g.selectAll(".agg-feature-layer")
        .on("mouseover", function() {
            if (state.isExpandLayer) return;
            d3.select(this)
                .select(".agg-feature-layer-frame")
                .style("opacity", 1);
            g.selectAll(".agg-link-path-fc").style("opacity", 1);
            g.selectAll(`[id^="feature-layer-frame-${sortedGNNFeatures.length-1}-node-"]`).style("opacity", 1);

        })
        .on("mouseout", function() {
            if (state.isExpandLayer) return;
            d3.select(this)
                .select(".agg-feature-layer-frame")
                .style("opacity", 0.5);
            g.selectAll(".agg-link-path-fc")
                .style("opacity", 0.1);
            g.selectAll(".feature-layer-frame")
                .style("opacity", 0.5);
        });
}

export function interactNodesAndActivateMatrixSubpipe(container: HTMLDivElement, selectedNode: number, adjacencyMatrix: number[][], mode: string) {
    const g = d3.select(container).select("svg");
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

export function interactNodesAndDeactivateMatrixSubpipe(container: HTMLDivElement) {
    const g = d3.select(container).select("svg");
    g.selectAll(".adj-matrix-row-border, .adj-matrix-col-border").style("opacity", 0);
}

export function interactLayerExpansionPipe(container: HTMLDivElement, cellWidth: number, adjacencyMatrix: number[][], sortedGNNFeatures: any[][], modelInfo: any, state: InteractionState){
    const g = d3.select(container);
    const maxLayerNum = sortedGNNFeatures.length - 1;

    g.selectAll(".feature-layer")
        .on("click", function(event: any) {
            event.stopPropagation();
            const id = (this as HTMLElement).id;
            const matchedID = extractFeatureId(id);
            if (!matchedID) return;
            const layerID = Number(matchedID[1]);
            const nodeID = Number(matchedID[2]);
            if (!Number.isInteger(layerID) || !Number.isInteger(nodeID)) return;
            if (layerID === 0) return;
            state.isExpandLayer = !state.isExpandLayer;
            state.currentLayerID = layerID;
            console.log("isExpandLayer click:", state.isExpandLayer, matchedID);
            g.selectAll(".link-path, .link-path-fc, .agg-link-path-fc").style("opacity", 0);
            g.selectAll(".feature-layer, .fc-feature-layer, .agg-feature-layer")
                .style("opacity", 0.1)
                .attr("pointer-events", "none");
            d3.select(this).style("opacity", 1);
            for(let i = 0; i < adjacencyMatrix[nodeID].length; i++) {
                if (adjacencyMatrix[nodeID][i] === 1){
                    g.select(`#feature-layer-${layerID-1}-node-${i}`).style("opacity", 1);
                }
            }
            const dist = 50 * 3 + sortedGNNFeatures[layerID-1][0].length * cellWidth + sortedGNNFeatures[layerID][0].length * cellWidth - 100;
            transitFeatureLayers(container, layerID, dist);
            let direction = "up";
            if (nodeID < (adjacencyMatrix.length)/2) direction = "down";
            visualizeInnerGNNLayerSubpipe(container, cellWidth, layerID, nodeID, adjacencyMatrix, sortedGNNFeatures, modelInfo, direction);
            resizeSvgToContent(container);
        });

    g.on("click", function() {
        if(state.isExpandLayer) state.isExpandLayer = false;
        console.log("isExpandLayer updated:", state.isExpandLayer);
        g.selectAll(".link-path, .link-path-fc, .agg-link-path-fc").style("opacity", 0.1);
        g.selectAll(".feature-layer, .fc-feature-layer, .agg-feature-layer")
            .style("opacity", 1)
            .attr("pointer-events", "auto");
        g.selectAll(".layer-inner-works-group").remove();
        if(state.currentLayerID !== -1 && state.currentLayerID != maxLayerNum+1)transitFeatureLayers(container, state.currentLayerID, 0);
        if(state.currentLayerID === maxLayerNum+1)transitFCLayer(container, 0);
        state.currentLayerID = -1;
        resizeSvgToContent(container);
    });
}

export function interactFCExpansionPipe(container: HTMLDivElement, cellWidth: number, sortedGNNFeatures: any[][], modelInfo: any, mode: string, state: InteractionState){
    const biasDim = 4;
    const g = d3.select(container);
    const maxLayerNum = sortedGNNFeatures.length - 1;
    g.selectAll(".fc-feature-layer")
        .on("click", function(event: any) {
            event.stopPropagation();
            state.isExpandLayer = true;
            state.currentLayerID = maxLayerNum + 1;
            const id = extractFCNodeIndex((this as HTMLElement).id);
            g.selectAll(".link-path, .link-path-fc, .agg-link-path-fc").style("opacity", 0);
            g.selectAll(".feature-layer, .fc-feature-layer, .agg-feature-layer")
                .style("opacity", 0.1)
                .attr("pointer-events", "none");
            d3.select(this).style("opacity", 1);
            if (mode != "graph") g.select("#feature-layer-" + maxLayerNum + "-node-" + id).style("opacity", 1);
            // the problem for both transition distance is need to minus the 'gap=100' to align the view!!!
            const transitDistance = (sortedGNNFeatures[sortedGNNFeatures.length-1][0].length + biasDim * 2) * cellWidth + distanceToFeature * 3 - 100;
            transitFCLayer(container, transitDistance);
            let direction = "down";
            if (id < (sortedGNNFeatures[sortedGNNFeatures.length-1].length)/2) direction = "up";
            visualizeInnerFCLayerSubpipe(container, cellWidth, id, sortedGNNFeatures, modelInfo, direction, mode);
            resizeSvgToContent(container);
        });
}
