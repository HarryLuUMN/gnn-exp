import * as d3 from 'd3';

export function interactNodesAndLinksPipe(adjacencyMatrix: number[][]) {
    const g = d3.select("#matvis");
    g.selectAll(".feature-layer")
        .on("mouseover", function(event: any, d: any) {
            const id = (this as HTMLElement).id;
            const match = id.match(/^feature-layer-(\d+)-node-(\d+)$/);

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
