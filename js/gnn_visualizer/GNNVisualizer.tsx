import React, { useEffect, useRef } from "react";
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
    
    useEffect(() => {
        const container = containerRef.current;
        if (!container || !intmData || !modelInfo || !graphData) {
            return;
        }

        console.log("modelInfo in GNNVisualizer:", modelInfo);
        console.log("intmData in GNNVisualizer:", intmData);
        console.log("graphData in GNNVisualizer:", graphData);
        console.log("queries in GNNVisualizer:", queries);
        console.log("mode in GNNVisualizer:", mode);

        modelPipeline(
            container,
            () => undefined,
            modelInfo,
            intmData,
            graphData,
            queries,
            subgraphSample,
            mode
        );
        onLoadComplete();
        return () => {
            container.replaceChildren();
        };
    }, [graphData, intmData, modelInfo, onLoadComplete, queries, renderToken, subgraphSample, mode]);
    

    return (
        <div
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
