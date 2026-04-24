import React, { useEffect, useRef, useState } from "react";
import {
    detectCapabilities,
    resolveRenderer,
    type RendererMode,
    type ResolvedRenderer,
} from "../renderers/capabilities";
import { modelPipeline } from "./modelPipeline";

const MIN_MODEL_ZOOM = 0.25;
const MAX_MODEL_ZOOM = 4;
const MODEL_ZOOM_STEP = 0.25;

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

type PanState = {
    x: number;
    y: number;
};

type DragState = {
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
    didMove: boolean;
};

function isFeatureExpansionTarget(target: EventTarget | null) {
    return (
        target instanceof Element &&
        !!target.closest(".feature-layer, .fc-feature-layer")
    );
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
    const viewportRef = useRef<HTMLDivElement>(null);
    const dragRef = useRef<DragState | null>(null);
    const suppressClickRef = useRef(false);
    const [modelZoom, setModelZoom] = useState(1);
    const [modelPan, setModelPan] = useState<PanState>({ x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState(false);
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

    const zoomOut = React.useCallback(() => {
        setModelZoom((value) => Math.max(MIN_MODEL_ZOOM, value - MODEL_ZOOM_STEP));
    }, []);

    const zoomIn = React.useCallback(() => {
        setModelZoom((value) => Math.min(MAX_MODEL_ZOOM, value + MODEL_ZOOM_STEP));
    }, []);

    const resetView = React.useCallback(() => {
        setModelZoom(1);
        setModelPan({ x: 0, y: 0 });
    }, []);

    const onPointerDown = React.useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) {
                return;
            }
            if (isFeatureExpansionTarget(event.target)) {
                return;
            }

            dragRef.current = {
                pointerId: event.pointerId,
                startX: event.clientX,
                startY: event.clientY,
                originX: modelPan.x,
                originY: modelPan.y,
                didMove: false,
            };
            event.currentTarget.setPointerCapture(event.pointerId);
        },
        [modelPan.x, modelPan.y]
    );

    const onPointerMove = React.useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            const drag = dragRef.current;
            if (!drag || drag.pointerId !== event.pointerId) {
                return;
            }

            const dx = event.clientX - drag.startX;
            const dy = event.clientY - drag.startY;
            if (!drag.didMove && Math.hypot(dx, dy) < 3) {
                return;
            }

            drag.didMove = true;
            setIsPanning(true);
            setModelPan({
                x: drag.originX + dx,
                y: drag.originY + dy,
            });
            event.preventDefault();
        },
        []
    );

    const finishPan = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        const drag = dragRef.current;
        if (!drag || drag.pointerId !== event.pointerId) {
            return;
        }

        if (drag.didMove) {
            suppressClickRef.current = true;
            window.setTimeout(() => {
                suppressClickRef.current = false;
            }, 0);
        }

        dragRef.current = null;
        setIsPanning(false);
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
        }
    }, []);

    const onClickCapture = React.useCallback(
        (event: React.MouseEvent<HTMLDivElement>) => {
            if (!suppressClickRef.current) {
                return;
            }

            event.preventDefault();
            event.stopPropagation();
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
        <div className="gnn-model-shell">
            <div className="gnn-model-toolbar">
                <div className="gnn-model-toolbar__status">
                    <span>
                        Requested: <strong>{renderer.toUpperCase()}</strong>
                    </span>
                    <span>
                        Active: <strong>{resolution.effectiveRenderer.toUpperCase()}</strong>
                    </span>
                    <span>Zoom: {Math.round(modelZoom * 100)}%</span>
                    {resolution.reason ? (
                        <span className="gnn-model-toolbar__reason">
                            {resolution.reason}
                        </span>
                    ) : null}
                </div>
                <div className="gnn-model-toolbar__controls">
                    <span className="gnn-model-toolbar__hint">
                        Drag the canvas to pan.
                    </span>
                    <button
                        className="gnn-model-button"
                        type="button"
                        onClick={zoomOut}
                        disabled={modelZoom <= MIN_MODEL_ZOOM}
                    >
                        Zoom Out
                    </button>
                    <button
                        className="gnn-model-button"
                        type="button"
                        onClick={resetView}
                        disabled={
                            modelZoom === 1 &&
                            modelPan.x === 0 &&
                            modelPan.y === 0
                        }
                    >
                        Reset
                    </button>
                    <button
                        className="gnn-model-button"
                        type="button"
                        onClick={zoomIn}
                        disabled={modelZoom >= MAX_MODEL_ZOOM}
                    >
                        Zoom In
                    </button>
                </div>
            </div>
            <div
                ref={viewportRef}
                className={`gnn-model-viewport${
                    isPanning ? " gnn-model-viewport--panning" : ""
                }`}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={finishPan}
                onPointerCancel={finishPan}
                onClickCapture={onClickCapture}
            >
                <div
                    ref={containerRef}
                    className="gnn-model-content"
                    style={{
                        transform: `translate(${modelPan.x}px, ${modelPan.y}px) scale(${modelZoom})`,
                    }}
                />
            </div>
        </div>
    );
};

export default GNNVisualizer;
