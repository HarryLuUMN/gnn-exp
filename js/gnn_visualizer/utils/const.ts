import * as d3 from "d3";

export const featureColor = d3
    .scaleLinear<string>()
    .domain([-3, -1, -0.1, 0, 0.1, 1, 3])
    .range(["#304E30", "#3DBA41", "#B7EFB8", "white", "#BBB7EF", "#6E09CD", "#4B0092"]);

export const curve = d3.line().curve(d3.curveBasis);