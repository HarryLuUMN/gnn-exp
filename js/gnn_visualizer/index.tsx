import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import "./style.css";
import GNNVisualizer from "./GNNVisualizer";
import type {
    RendererMode,
    ResolvedRenderer,
} from "../renderers/capabilities";

const render = createRender(() => {
    const [graphData] = useModelState<any>("graphData");
    const [intmData] = useModelState<any>("intmData");
    const [modelInfo] = useModelState<any>("modelInfo");
    const [queries] = useModelState<number[][]>("queries");
    const [subgraphSample] = useModelState<any>("subgraphSample");
    const [renderToken] = useModelState<number>("renderToken");
    const [mode] = useModelState<string>("mode");
    const [renderer] = useModelState<RendererMode>("renderer");
    const [effectiveRenderer, setEffectiveRenderer] =
        useModelState<ResolvedRenderer>("effectiveRenderer");
    const [viewportHeight, setViewportHeight] =
        useModelState<number>("viewportHeight");
    const [autoFit] = useModelState<boolean>("autoFit");

	console.log("GNNVisualizer received graphData =", graphData);
	console.log("GNNVisualizer received queries =", queries, "type:", typeof queries, "isArray:", Array.isArray(queries));
	console.log("GNNVisualizer received mode =", mode);

    return (
        <div className="gnn_vis_widgets">
            <GNNVisualizer
                graphData={graphData}
                modelInfo={modelInfo}
                onLoadComplete={() => undefined}
                intmData={intmData}
                renderToken={renderToken}
                queries={queries}
                subgraphSample={subgraphSample}
                mode={mode}
                renderer={renderer}
                effectiveRenderer={effectiveRenderer}
                setEffectiveRenderer={setEffectiveRenderer}
                viewportHeight={viewportHeight}
                setViewportHeight={setViewportHeight}
                autoFit={autoFit}
            />
        </div>
    );
});

export default { render };
