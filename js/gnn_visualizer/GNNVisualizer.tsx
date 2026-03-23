import React, { useEffect, useRef, useState } from "react";
import { modelPipeline } from "./modelPipeline";

interface GNNVisualizerProps {
    intmData: any;
    modelInfo: any;
    onLoadComplete: () => void;
    graphData: any;
    renderToken: number;
    queries: number[][];
    subgraphSample: any;
    mode: string;
}

const GNNVisualizer: React.FC<GNNVisualizerProps> = ({
    intmData,
    modelInfo, 
    onLoadComplete,
    graphData,
    renderToken,
    queries,
    subgraphSample,
    mode,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isLoading, setIsLoading] = useState(true);

    if (intmData != null) {

    }
    
    useEffect(() => {
        console.log("modelInfo in GNNVisualizer:", modelInfo);
        console.log("intmData in GNNVisualizer:", intmData);
        console.log("graphData in GNNVisualizer:", graphData);
        console.log("queries in GNNVisualizer:", queries);
        console.log("mode in GNNVisualizer:", mode);
        modelPipeline(setIsLoading, modelInfo, intmData, graphData, queries, subgraphSample, mode);
        onLoadComplete();
    }, [graphData, intmData, renderToken, queries, mode]);
    

    return (
        <div
            id="matvis"
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
