import React, { useEffect, useRef, useState } from "react";
import { visualizationPipeline } from "./visualizationPipeline";

interface GNNVisualizerProps {
    intmData: any;
    modelInfo: any;
    onLoadComplete: () => void;
    graphData: any;
}

const GNNVisualizer: React.FC<GNNVisualizerProps> = ({
    intmData,
    modelInfo, 
    onLoadComplete,
    graphData,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [isLoading, setIsLoading] = useState(true);

    if (intmData != null) {

    }
    
    useEffect(() => {
           visualizationPipeline(setIsLoading, modelInfo, intmData, graphData);
           onLoadComplete();
    }, [graphData, intmData]);
    

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
