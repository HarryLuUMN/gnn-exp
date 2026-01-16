import React, { useEffect, useRef, useState } from "react";
import { modelPipeline } from "./modelPipeline";
import { initCanvasId, initSvgId } from "../states";

interface GNNVisualizerProps {
    intmData: any;
    modelInfo: any;
    onLoadComplete: () => void;
    graphData: any;
    renderToken: number;
    queries: number[][];
    subgraphSample: any;
}

const GNNVisualizer: React.FC<GNNVisualizerProps> = ({
    intmData,
    modelInfo, 
    onLoadComplete,
    graphData,
    renderToken,
    queries,
    subgraphSample
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isLoading, setIsLoading] = useState(true);
    const canvasId = initCanvasId();
    const svgId = initSvgId();

    if (intmData != null) {

    }
    
    useEffect(() => {
        console.log("modelInfo in GNNVisualizer:", modelInfo);
        console.log("intmData in GNNVisualizer:", intmData);
        console.log("graphData in GNNVisualizer:", graphData);
        console.log("queries in GNNVisualizer:", queries);
        modelPipeline(setIsLoading, modelInfo, intmData, graphData, [[12, 18]], subgraphSample);
        onLoadComplete();
    }, [graphData, intmData, renderToken, queries]);
    

    return (
        <div
            id={canvasId}
            ref={containerRef}
            style={{
                width: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "start",
                height: "auto",
                overflow: "auto", 
                overflowX: "auto",
            }}
        ></div>
    );
};

export default GNNVisualizer;
