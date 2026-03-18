import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import "./style.css";
import GraphEditor from "./GraphEditor";

const render = createRender(() => {

    const [dataFile] = useModelState<string>("dataFile");
    const [, setGraphData] = useModelState<any>("graphData");  
    const [, setNodePositions] = React.useState<any[]>([]);

    const handleSimulatedGraphChange = (value: any) => {
        setGraphData(value);  
        console.log("Graph data updated:", value);
    };

    const handleNodePositionsChange = (positions: { id: string; x: number; y: number }[]) => {
        setNodePositions(positions);
        console.log("Node positions updated:", positions);
    };
	
	return (
		<div className="graph_editor">
			<GraphEditor
                dataFile={dataFile}
                handleSimulatedGraphChange={handleSimulatedGraphChange}
                onNodePositionsChange={handleNodePositionsChange}
            />
		</div>
	);
});

export default { render };
