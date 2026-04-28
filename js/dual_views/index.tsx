import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import "./style.css";
import DualViews from "./DualViews";
import type {
    RendererMode,
    ResolvedRenderer,
} from "./renderers/capabilities";

const render = createRender(() => {
    const [graphData] = useModelState<any>("graphData");
    const [renderer] = useModelState<RendererMode>("renderer");
    const [effectiveRenderer, setEffectiveRenderer] =
        useModelState<ResolvedRenderer>("effectiveRenderer");

	console.log("DualViews received graphData =", graphData);

    return (
        <div className="gnn_vis_widgets">
            <DualViews
                renderer={renderer}
                effectiveRenderer={effectiveRenderer}
                setEffectiveRenderer={setEffectiveRenderer}
                graphData={graphData}
                handleSimulatedGraphChange={() => {}}
                handleNodePositionsChange={() => {}}
            />
        </div>
    );
});

export default { render };
