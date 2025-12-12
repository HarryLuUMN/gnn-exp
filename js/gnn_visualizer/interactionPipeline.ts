import * as d3 from 'd3';

export function interactNodesAndLinks(adjacencyMatrix: number[][]) {
    const g = d3.select("#matvis");
    g.selectAll(".feature-layer")
        .on("mouseover", function(event: any, d: any) {
            const id = (this as HTMLElement).id;
            const match = id.match(/^feature-layer-(\d+)-node-(\d+)$/);

            if (!match) return;

            const ithLayer = Number(match[1]);
            const jthNode = Number(match[2]);

            const neighbors: number[] = adjacencyMatrix[jthNode];

            console.log("mouseover node:", this, "layer:", ithLayer, "node:", jthNode);
            d3.select(this).select(".feature-layer-frame")
                .style("opacity", 1)
                .style("stroke", "black");
            
            for (let i=0; i<neighbors.length; i++){
                if (neighbors[i] === 0) continue;
                g.select(`#link-path-${ithLayer-1}-${i}-to-${jthNode}`)
                    .style("stroke", "black")
                    .style("opacity", 1);
            }
        })
        .on("mouseout", function(event: any, d: any) {
            console.log("mouseout node:", this);
            g.selectAll(".link-path")
                .style("stroke", "black")
                .style("opacity", 0.1);

            d3.select(this).select(".feature-layer-frame")
                .style("opacity", 0.5)
                .style("stroke", "black");
        });
}




