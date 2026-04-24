import React, { useEffect, useRef, useState } from "react";
import {
    detectCapabilities,
    resolveRenderer,
    type RendererMode,
    type ResolvedRenderer,
} from "../renderers/capabilities";
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
    renderer: RendererMode;
    effectiveRenderer: ResolvedRenderer;
    setEffectiveRenderer: (renderer: ResolvedRenderer) => void;
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
    renderer,
    effectiveRenderer,
    setEffectiveRenderer,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [fallbackFailures, setFallbackFailures] = useState<
        Partial<Record<ResolvedRenderer, string>>
    >({});
    const capabilities = React.useMemo(() => detectCapabilities(), []);
    const resolution = React.useMemo(
        () => resolveRenderer(renderer, capabilities, fallbackFailures),
        [capabilities, fallbackFailures, renderer]
    );

    useEffect(() => {
        if (effectiveRenderer !== resolution.effectiveRenderer) {
            setEffectiveRenderer(resolution.effectiveRenderer);
        }
    }, [effectiveRenderer, resolution.effectiveRenderer, setEffectiveRenderer]);

    const onBackendFailure = React.useCallback(
        (backend: Exclude<ResolvedRenderer, "svg">, reason: string) => {
            setFallbackFailures((current) => {
                if (current[backend] === reason) {
                    return current;
                }

                return { ...current, [backend]: reason };
            });
        },
        []
    );
    
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
        console.log("renderer in GNNVisualizer:", renderer, resolution);

        let cancelled = false;
        let cleanup: (() => void) | null = null;

        modelPipeline(
            container,
            () => undefined,
            modelInfo,
            intmData,
            graphData,
            queries,
            subgraphSample,
            mode,
            resolution.effectiveRenderer,
            (backend, reason) => {
                if (!cancelled) {
                    onBackendFailure(backend, reason);
                }
            }
        ).then((pipelineCleanup) => {
            if (cancelled) {
                pipelineCleanup?.();
                return;
            }

            cleanup = pipelineCleanup;
            if (pipelineCleanup || resolution.effectiveRenderer === "svg") {
                onLoadComplete();
            }
        }).catch((error: unknown) => {
            if (cancelled) {
                return;
            }

            const reason =
                error instanceof Error ? error.message : "Renderer failed unexpectedly.";
            if (resolution.effectiveRenderer === "svg") {
                console.error("SVG visualization pipeline failed.", error);
                onLoadComplete();
                return;
            }

            onBackendFailure(resolution.effectiveRenderer, reason);
        });

        return () => {
            cancelled = true;
            cleanup?.();
            container.replaceChildren();
        };
    }, [
        graphData,
        intmData,
        modelInfo,
        mode,
        onBackendFailure,
        onLoadComplete,
        queries,
        renderToken,
        renderer,
        resolution,
        subgraphSample,
    ]);
    

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
