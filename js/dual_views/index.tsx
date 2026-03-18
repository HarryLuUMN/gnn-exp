import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import "./style.css";
import DualViews from "./DualViews";

const render = createRender(() => {
    // Bind to Python trait `graphData`
    const [graphData] = useModelState<any>("graphData");

	console.log("DualViews received graphData =", graphData);

    return (
        <div className="gnn_vis_widgets">
            <DualViews
                graphData={graphData}
                handleSimulatedGraphChange={() => {}}
                handleNodePositionsChange={() => {}}
            />
        </div>
    );
});

export default { render };
