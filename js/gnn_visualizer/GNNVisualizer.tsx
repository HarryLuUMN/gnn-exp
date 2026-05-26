import React, { useEffect, useRef, useState } from "react";
import {
    detectCapabilities,
    detectWebgpuCapability,
    resolveRenderer,
    type RendererMode,
    type ResolvedRenderer,
} from "../renderers/capabilities";
import { modelPipeline } from "./modelPipeline";

const MIN_MODEL_ZOOM = 0.05;
const MAX_MODEL_ZOOM = 8;
const MODEL_ZOOM_SLIDER_STEP = 0.05;
const SWIPE_PAN_SPEED = 1;
const DEFAULT_VIEWPORT_HEIGHT = 820;
const MIN_VIEWPORT_HEIGHT = 420;
const MAX_VIEWPORT_HEIGHT = 1400;
const VIEWPORT_HEIGHT_STEP = 20;
const FIT_PADDING = 48;

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
    viewportHeight?: number;
    setViewportHeight?: (height: number) => void;
    autoFit?: boolean;
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

function isSvgVisualizationTarget(target: EventTarget | null) {
    return target instanceof Element && !!target.closest("svg");
}

function clamp(value: number, min: number, max: number) {
    return Math.min(max, Math.max(min, value));
}

function clampViewportHeight(value: number | undefined) {
    return clamp(
        Number.isFinite(value) ? Number(value) : DEFAULT_VIEWPORT_HEIGHT,
        MIN_VIEWPORT_HEIGHT,
        MAX_VIEWPORT_HEIGHT
    );
}

function numericAttribute(element: Element, name: string) {
    const raw = element.getAttribute(name);
    if (!raw) {
        return 0;
    }

    const value = Number(raw);
    return Number.isFinite(value) ? value : 0;
}

function measureContentSize(container: HTMLDivElement) {
    const child = container.firstElementChild;
    let width = container.scrollWidth;
    let height = container.scrollHeight;

    if (child instanceof SVGSVGElement) {
        width = Math.max(
            width,
            child.width.baseVal.value,
            numericAttribute(child, "width"),
            child.viewBox.baseVal.width
        );
        height = Math.max(
            height,
            child.height.baseVal.value,
            numericAttribute(child, "height"),
            child.viewBox.baseVal.height
        );
    } else if (child instanceof HTMLElement) {
        width = Math.max(width, child.offsetWidth, child.scrollWidth);
        height = Math.max(height, child.offsetHeight, child.scrollHeight);
    }

    return { width, height };
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
    viewportHeight,
    setViewportHeight,
    autoFit = true,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const viewportRef = useRef<HTMLDivElement>(null);
    const dragRef = useRef<DragState | null>(null);
    const pipelineRunRef = useRef(0);
    const suppressClickRef = useRef(false);
    const fitFrameRef = useRef<number | null>(null);
    const fitTimeoutRef = useRef<number | null>(null);
    const [modelZoom, setModelZoom] = useState(1);
    const [modelPan, setModelPan] = useState<PanState>({ x: 0, y: 0 });
    const [localViewportHeight, setLocalViewportHeight] = useState(
        DEFAULT_VIEWPORT_HEIGHT
    );
    const [isPanning, setIsPanning] = useState(false);
    const [fallbackFailures, setFallbackFailures] = useState<
        Partial<Record<ResolvedRenderer, string>>
    >({});
    const [capabilities, setCapabilities] = useState(() => detectCapabilities());
    const resolution = React.useMemo(
        () => resolveRenderer(renderer, capabilities, fallbackFailures),
        [capabilities, fallbackFailures, renderer]
    );
    const resolvedViewportHeight = clampViewportHeight(
        viewportHeight ?? localViewportHeight
    );

    const updateViewportHeight = React.useCallback(
        (height: number) => {
            const nextHeight = clampViewportHeight(height);
            if (setViewportHeight) {
                setViewportHeight(nextHeight);
                return;
            }

            setLocalViewportHeight(nextHeight);
        },
        [setViewportHeight]
    );

    useEffect(() => {
        let cancelled = false;

        detectWebgpuCapability().then((webgpu) => {
            if (cancelled) {
                return;
            }

            setCapabilities((current) =>
                current.webgpu === webgpu ? current : { ...current, webgpu }
            );
        });

        return () => {
            cancelled = true;
        };
    }, []);

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

    const resetView = React.useCallback(() => {
        setModelZoom(1);
        setModelPan({ x: 0, y: 0 });
    }, []);

    const fitContentToViewport = React.useCallback(() => {
        const viewport = viewportRef.current;
        const container = containerRef.current;
        if (!viewport || !container) {
            return;
        }

        const contentSize = measureContentSize(container);
        if (contentSize.width <= 0 || contentSize.height <= 0) {
            return;
        }

        const availableWidth = Math.max(1, viewport.clientWidth - FIT_PADDING);
        const availableHeight = Math.max(1, viewport.clientHeight - FIT_PADDING);
        const nextZoom = clamp(
            Math.min(1, availableWidth / contentSize.width, availableHeight / contentSize.height),
            MIN_MODEL_ZOOM,
            MAX_MODEL_ZOOM
        );

        setModelZoom(nextZoom);
        setModelPan({
            x: Math.max(0, (viewport.clientWidth - contentSize.width * nextZoom) / 2),
            y: Math.max(0, (viewport.clientHeight - contentSize.height * nextZoom) / 2),
        });
    }, []);

    const clearScheduledFit = React.useCallback(() => {
        if (fitFrameRef.current != null) {
            window.cancelAnimationFrame(fitFrameRef.current);
            fitFrameRef.current = null;
        }

        if (fitTimeoutRef.current != null) {
            window.clearTimeout(fitTimeoutRef.current);
            fitTimeoutRef.current = null;
        }
    }, []);

    const scheduleFitToContent = React.useCallback(() => {
        clearScheduledFit();
        fitFrameRef.current = window.requestAnimationFrame(() => {
            fitFrameRef.current = null;
            fitContentToViewport();
        });
        fitTimeoutRef.current = window.setTimeout(() => {
            fitTimeoutRef.current = null;
            fitContentToViewport();
        }, 650);
    }, [clearScheduledFit, fitContentToViewport]);

    const onViewportHeightChange = React.useCallback(
        (event: React.ChangeEvent<HTMLInputElement>) => {
            updateViewportHeight(Number(event.currentTarget.value));
        },
        [updateViewportHeight]
    );

    const onModelZoomChange = React.useCallback(
        (event: React.ChangeEvent<HTMLInputElement>) => {
            setModelZoom(clamp(
                Number(event.currentTarget.value),
                MIN_MODEL_ZOOM,
                MAX_MODEL_ZOOM
            ));
        },
        []
    );

    const onPointerDown = React.useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) {
                return;
            }
            if (isFeatureExpansionTarget(event.target)) {
                return;
            }
            if (
                resolution.effectiveRenderer === "svg" &&
                isSvgVisualizationTarget(event.target)
            ) {
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
        [modelPan.x, modelPan.y, resolution.effectiveRenderer]
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

    const onWheel = React.useCallback((event: WheelEvent) => {
        const horizontalDelta = event.shiftKey ? event.deltaY : event.deltaX;
        if (Math.abs(horizontalDelta) < 1) {
            return;
        }
        if (!event.shiftKey && Math.abs(horizontalDelta) < Math.abs(event.deltaY)) {
            return;
        }

        event.preventDefault();
        setModelPan((current) => ({
            ...current,
            x: current.x - horizontalDelta * SWIPE_PAN_SPEED,
        }));
    }, []);

    useEffect(() => {
        const viewport = viewportRef.current;
        if (!viewport) {
            return;
        }

        viewport.addEventListener("wheel", onWheel, { passive: false });
        return () => {
            viewport.removeEventListener("wheel", onWheel);
        };
    }, [onWheel]);

    useEffect(() => {
        return () => {
            clearScheduledFit();
        };
    }, [clearScheduledFit]);

    useEffect(() => {
        if (autoFit) {
            scheduleFitToContent();
        }
    }, [autoFit, resolvedViewportHeight, scheduleFitToContent]);
    
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

        const runId = pipelineRunRef.current + 1;
        pipelineRunRef.current = runId;
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
                if (!cancelled && runId === pipelineRunRef.current) {
                    onBackendFailure(backend, reason);
                }
            }
        ).then((pipelineCleanup) => {
            if (cancelled || runId !== pipelineRunRef.current) {
                return;
            }

            cleanup = pipelineCleanup;
            if (autoFit) {
                scheduleFitToContent();
            }
            if (pipelineCleanup || resolution.effectiveRenderer === "svg") {
                onLoadComplete();
            }
        }).catch((error: unknown) => {
            if (cancelled || runId !== pipelineRunRef.current) {
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
            if (runId === pipelineRunRef.current) {
                cleanup?.();
                container.replaceChildren();
            }
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
        autoFit,
        scheduleFitToContent,
        subgraphSample,
    ]);
    

    return (
        <div className="gnn-model-shell">
            <div className="gnn-model-toolbar">
                <div className="gnn-model-toolbar__status">
                    <span>
                        Renderer: <strong>{resolution.effectiveRenderer.toUpperCase()}</strong>
                    </span>
                    {renderer !== "auto" ? (
                        <span>
                            API: <strong>{renderer.toUpperCase()}</strong>
                        </span>
                    ) : null}
                    <span>Zoom: {Math.round(modelZoom * 100)}%</span>
                    <span>Viewport: {resolvedViewportHeight}px</span>
                    {resolution.reason ? (
                        <span className="gnn-model-toolbar__reason">
                            {resolution.reason}
                        </span>
                    ) : null}
                </div>
                <div className="gnn-model-toolbar__controls">
                    <label className="gnn-model-toolbar__range">
                        <span>Height</span>
                        <input
                            aria-label="Model viewport height"
                            className="gnn-model-height-slider"
                            max={MAX_VIEWPORT_HEIGHT}
                            min={MIN_VIEWPORT_HEIGHT}
                            onChange={onViewportHeightChange}
                            step={VIEWPORT_HEIGHT_STEP}
                            type="range"
                            value={resolvedViewportHeight}
                        />
                    </label>
                    <label className="gnn-model-toolbar__range">
                        <span>Zoom</span>
                        <input
                            aria-label="Model zoom"
                            className="gnn-model-zoom-slider"
                            max={MAX_MODEL_ZOOM}
                            min={MIN_MODEL_ZOOM}
                            onChange={onModelZoomChange}
                            step={MODEL_ZOOM_SLIDER_STEP}
                            type="range"
                            value={modelZoom}
                        />
                    </label>
                    <button
                        className="gnn-model-button"
                        type="button"
                        onClick={fitContentToViewport}
                    >
                        Fit
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
                </div>
            </div>
            <div
                ref={viewportRef}
                className={`gnn-model-viewport${
                    isPanning ? " gnn-model-viewport--panning" : ""
                }`}
                style={{ height: `${resolvedViewportHeight}px` }}
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
