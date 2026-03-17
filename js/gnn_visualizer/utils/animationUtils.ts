import * as d3 from "d3";

export function transitFeatureLayers(container: HTMLDivElement, layerID: number, distanceX: number) {
    d3.select(container).selectAll(".feature-layer")
        .filter(function () {
            const idStr = (this as HTMLElement).id;
            const match = idStr.match(/^feature-layer-(\d+)-node-\d+$/);
            if (!match) return false;
            const id = Number(match[1]);
            return id >= layerID;
        }).transition().duration(500).ease(d3.easeCubicOut).attr("transform", `translate(${distanceX}, 0)`);
    transitFCLayer(container, distanceX);
}

export function transitFCLayer(container: HTMLDivElement, distanceX: number) {
    d3.select(container).selectAll(".fc-feature-layer").transition().duration(500).ease(d3.easeCubicOut).attr("transform", `translate(${distanceX}, 0)`);
}
