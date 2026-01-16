import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import "./style.css";
import GNNVisualizer from "./GNNVisualizer";

const render = createRender(() => {
    // Bind to Python trait `graphData`
    const [graphData, setGraphData] = useModelState<any>("graphData");
    const [graphPath, setGraphPath] = useModelState<string>("graphPath");
    const [intmData, setIntmData] = useModelState<any>("intmData");
    const [modelInfo, setModelInfo] = useModelState<any>("modelInfo");

    const [queries, setQueries] = useModelState<number[][]>("queries");
    const [subgraphSample, setSubgraphSample] = useModelState<any>("subgraphSample");
    const [renderToken] = useModelState<number>("renderToken");

    const [isLoading, setIsLoading] = React.useState(true);

	console.log("GNNVisualizer received graphData =", graphData);
	console.log("GNNVisualizer received queries =", queries, "type:", typeof queries, "isArray:", Array.isArray(queries));

    return (
        <div className="gnn_vis_widgets">
            <GNNVisualizer
                graphData={graphData}
                modelInfo={modelInfo}
                onLoadComplete={() =>setIsLoading(false)}
                intmData={intmData}
                renderToken={renderToken}
                queries={queries}
                subgraphSample={subgraphSample}
            />
        </div>
    );
});

export default { render };
