import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import {
    loadSimGraphData,
    parseFeatureText,
    processDataFromEditorToVisualizer,
    processDataFromVisualizerToEditor,
    randomizeFeatures,
} from "./graphEditorUtils";

interface GraphEditorProps {
    dataFile: any;
    initialGraphData?: any;
    handleSimulatedGraphChange?: (value: any) => void;
    onNodePositionsChange?: (nodePositions: { id: string; x: number; y: number }[]) => void;
}

export default function GraphEditor({
    dataFile, 
    initialGraphData,
    handleSimulatedGraphChange,
    onNodePositionsChange,
}: GraphEditorProps): React.ReactElement {
    const [, setIsRunning] = useState(true);

    const svgContainer = useRef<HTMLDivElement>(null);
    const simulationRef = useRef<d3.Simulation<any, any> | null>(null);
    const isRunningRef = useRef(true);
    const isDraggingRef = useRef(false);
    const linksRef = useRef<any[]>([]);
    const nodesRef = useRef<any[]>([]);
    const selectionState = useRef(false);
    const loadedDataFileRef = useRef<string | null>(null);

    const datasetRef = useRef<any>({});
    const [mode, setMode] = useState("edge");
    const modeRef = useRef("edge");

    // Sync state
    const selectedNodeRef = useRef<string | null>(null);
    const secondSelectedNodeRef = useRef<string | null>(null);
    const [, setSelectedNodeId] = useState<string | null>(null);
    const [, setSecondSelectedNodeId] = useState<
        string | null
    >(null);

    const [simGraphData, setSimGraphData] = React.useState<any>();

    const [feature, setFeature] = useState<string>("");
    const [featureDim, setFeatureDim] = useState<number>(0);


    const selectedLinkRef = useRef<SVGLineElement | null>(null);

    const clearEditorSelection = () => {
        if (svgContainer.current) {
            d3.select(svgContainer.current)
                .selectAll("circle")
                .attr("stroke", "#aaa");
            d3.select(svgContainer.current)
                .selectAll("line")
                .attr("stroke", "#aaa");
        }
        selectedNodeRef.current = null;
        secondSelectedNodeRef.current = null;
        selectedLinkRef.current = null;
        setSelectedNodeId(null);
        setSecondSelectedNodeId(null);
        selectionState.current = false;
    };

    const getCurrentDataset = () => ({
        nodes: nodesRef.current,
        links: linksRef.current,
    });

    const getEndpointId = (endpoint: any) => {
        if (typeof endpoint === "string") return endpoint;
        return endpoint?.id;
    };

    const getNextNodeId = () => {
        let maxNodeIndex = -1;
        for (const node of nodesRef.current) {
            const match = typeof node.id === "string" ? node.id.match(/^N(\d+)$/) : null;
            if (!match) continue;
            maxNodeIndex = Math.max(maxNodeIndex, Number(match[1]));
        }
        return `N${maxNodeIndex + 1}`;
    };

    useEffect(() => {
        if (!dataFile && !initialGraphData) return;
        if (loadedDataFileRef.current === dataFile) return;

        console.log("Loading JSON:", dataFile);

        const applyGraphData = (data: any) => {
            loadedDataFileRef.current = dataFile;
            setSimGraphData(data);

            let dim = 34;
            if (data.x && data.x.length > 0) {
                const first = data.x[0];
                if (Array.isArray(first)) {
                    dim = first.length;
                }
            }

            const featureText = randomizeFeatures(dim);
            setFeature(featureText);
            setFeatureDim(dim);
            console.log("Feature dimension:", dim, featureText);
        };

        if (initialGraphData?.x && initialGraphData?.edge_index) {
            applyGraphData(initialGraphData);
            return;
        }

        void loadSimGraphData(dataFile)
            .then((data) => {
                console.log("Loaded JSON:", data);
                applyGraphData(data);
            })
            .catch((err) => {
                console.error("Error loading JSON:", err);
                if (initialGraphData?.x && initialGraphData?.edge_index) {
                    applyGraphData(initialGraphData);
                }
            });

        
    }, [dataFile, initialGraphData]);


    useEffect(() => {
        console.log("Data file path:", dataFile);
        if (!simGraphData) return;    

        if (!svgContainer.current) return;

        const width = 640;
        const height = 640;
        let data = simGraphData;
                console.log("Loaded graph:", data);

                const initialData = processDataFromVisualizerToEditor(data);
                datasetRef.current = initialData;

                const links = initialData.links.map((d) => Object.create(d));
                const nodes = initialData.nodes.map((d) => ({ ...d }));

                nodesRef.current = nodes;
                linksRef.current = links;

                const simulation = d3
                    .forceSimulation<any>(nodes)
                    .force(
                        "link",
                        d3
                            .forceLink(links)
                            .id((d: any) => d.id)
                            .distance(100)
                    )
                    .force("charge", d3.forceManyBody().strength(-100))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .on("tick", ticked);

                simulationRef.current = simulation;

                const svg = d3
                    .create("svg")
                    .attr("width", width)
                    .attr("height", height)
                    .attr("viewBox", [0, 0, width, height])
                    .attr("style", "max-width: 100%; height: auto;")
                    .on("click", (event: MouseEvent) => {
                        if (!isRunningRef.current && modeRef.current === "node") {
                            if (selectedLinkRef.current) {
                                d3.select(selectedLinkRef.current).attr(
                                    "stroke",
                                    "#999"
                                );
                                selectedLinkRef.current = null;
                            }

                            if (!selectionState.current) {
                                const point = d3.pointer(event);
                                addNodeAt(point[0], point[1]);
                            } else {
                                clearEditorSelection();
                            }
                            return;
                        }

                        if (selectionState.current || selectedLinkRef.current) {
                            clearEditorSelection();
                        }
                    });

                svgContainer.current.innerHTML = "";
                svgContainer.current.appendChild(svg.node()!);

                const linkGroup = svg
                    .append("g")
                    .attr("stroke", "#aaa")
                    .attr("stroke-opacity", 0.6);

                const nodeGroup = svg
                    .append("g")
                    .attr("stroke", "#fff")
                    .attr("stroke-width", 1.5);

                const labelGroup = svg
                    .append("g")
                    .attr("font-family", "sans-serif")
                    .attr("font-size", 12)
                    .attr("text-anchor", "middle")
                    .attr("pointer-events", "none");

                function getNodeNumber(d: any) {
                    if (typeof d.id === "string") {
                    const m = d.id.match(/\d+/);
                    if (m) return +m[0];
                    }
                    return (d.index ?? nodesRef.current.indexOf(d)) + 1;
                }

                function ticked() {
                    linkGroup
                        .selectAll("line")
                        .data(linksRef.current)
                        .join("line")
                        .attr("x1", (d: any) => d.source.x)
                        .attr("y1", (d: any) => d.source.y)
                        .attr("x2", (d: any) => d.target.x)
                        .attr("y2", (d: any) => d.target.y)
                        .on("click", function (event: MouseEvent) {
                            event.stopPropagation();
                            if (modeRef.current !== "edge") return;

                            if (selectedLinkRef.current) {
                                d3.select(selectedLinkRef.current).attr(
                                    "stroke",
                                    "#aaa"
                                );
                            }

                            if (selectedLinkRef.current === this) {
                                selectedLinkRef.current = null;
                            } else {
                                d3.select(this).attr("stroke", "black");
                                selectedLinkRef.current =
                                    this as SVGLineElement;
                            }
                        });

                    nodeGroup
                        .selectAll("circle")
                        .data(nodesRef.current, (d: any) => d.id)
                        .join("circle")
                        .attr("r", 12)
                        .attr("stroke", "#aaa")
                        .attr("fill", "white")
                        .call(drag(simulation) as any)
                        .attr("cx", (d: any) => d.x)
                        .attr("cy", (d: any) => d.y)
                        .on("click", function (event: MouseEvent, d: any) {
                            event.stopPropagation();

                            const clickedId = d.id;

                            const isFirstSelected =
                                selectedNodeRef.current === clickedId;

                            if (modeRef.current === "node") {
                                clearEditorSelection();
                                if (!isFirstSelected) {
                                    d3.select(this).attr("stroke", "black");
                                    selectedNodeRef.current = clickedId;
                                    setSelectedNodeId(clickedId);
                                    selectionState.current = true;
                                }
                                return;
                            }

                            if (
                                selectedNodeRef.current &&
                                secondSelectedNodeRef.current
                            ) {
                                clearEditorSelection();
                                return;
                            }

                            if (!selectionState.current) {
                                d3.select(this).attr("stroke", "black");
                                selectedNodeRef.current = clickedId;
                                setSelectedNodeId(clickedId);
                                selectionState.current = true;
                                return;
                            }

                            if (
                                selectionState.current &&
                                !secondSelectedNodeRef.current &&
                                isFirstSelected
                            ) {
                                d3.select(this).attr("stroke", "#aaa");
                                selectedNodeRef.current = null;
                                setSelectedNodeId(null);
                                selectionState.current = false;
                                return;
                            }

                            if (
                                selectionState.current &&
                                !secondSelectedNodeRef.current &&
                                !isFirstSelected
                            ) {
                                d3.select(this).attr("stroke", "black");
                                secondSelectedNodeRef.current = clickedId;
                                setSecondSelectedNodeId(clickedId);

                                const sourceId = selectedNodeRef.current!;
                                const targetId = secondSelectedNodeRef.current!;

                                const alreadyLinked = linksRef.current.some(
                                    (link: any) =>
                                        (link.source.id === sourceId &&
                                            link.target.id === targetId) ||
                                        (link.source.id === targetId &&
                                            link.target.id === sourceId)
                                );

                                if (alreadyLinked) {
                                    clearEditorSelection();
                                } else {
                                    linksRef.current.push({
                                        source: sourceId,
                                        target: targetId,
                                        value: 1,
                                    });
                                    const linkForce =
                                        simulationRef.current?.force(
                                            "link"
                                        ) as d3.ForceLink<any, any>;
                                    linkForce?.links(linksRef.current);
                                    simulationRef.current?.alpha(0.5).restart();
                                    clearEditorSelection();
                                    handleTransmitToMainVisualizer();
                                }

                                return;
                            }
                        });
                    

                    labelGroup
                        .selectAll("text")
                        .data(nodesRef.current, (d: any) => d.id)
                        .join(
                        (enter) =>
                        enter
                        .append("text")
                        .text((d: any) => getNodeNumber(d))
                        .attr("dy", "0.35em")
                        .attr("stroke", "#fff")
                        .attr("stroke-width", 3)
                        .attr("paint-order", "stroke")
                        .attr("fill", "#111"),
                        (update) => update.text((d: any) => getNodeNumber(d))
                        )
                        .attr("x", (d: any) => d.x)
                        .attr("y", (d: any) => d.y);

                    if (onNodePositionsChange) {
                        onNodePositionsChange(nodesRef.current.map(node => ({ id: node.id, x: node.x, y: node.y })));
                    }
                }

                function drag(simulation: any) {
                    function dragstarted(event: any) {
                        isDraggingRef.current = true;
                        simulation.alphaTarget(0.3).restart();
                        event.subject.fx = event.subject.x;
                        event.subject.fy = event.subject.y;
                    }

                    function dragged(event: any) {
                        event.subject.fx = event.x;
                        event.subject.fy = event.y;
                    }

                    function dragended(event: any) {
                        isDraggingRef.current = false;
                        simulation.alphaTarget(0);
                        if (isRunningRef.current) {
                            event.subject.fx = null;
                            event.subject.fy = null;
                        }
                        handleTransmitToMainVisualizer();
                    }

                    return d3
                        .drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended);
                }

                function addNodeAt(x: number, y: number) {
                    const newNodeId = getNextNodeId();
                    // const featureText = randomizeFeatures(dim);

                    console.log("new node feature - before add 0:", feature);
                    console.log("feature dim:", featureDim);

                    const featureAdd = parseFeatureText(feature);

                    console.log("new node feature - before add 1:", featureAdd);

                    const newNode = {
                        id: newNodeId,
                        group: 3,
                        x,
                        y,
                        fx: x,
                        fy: y,
                        feature: featureAdd,
                    };
                    const newFeatureText = randomizeFeatures(featureDim);
                    setFeature(newFeatureText);

                    console.log("New node feature:", featureAdd, newFeatureText);
                    console.log("new node feature text:", feature);

                    console.log("Adding new node:", getCurrentDataset());
                    nodesRef.current.push(newNode);
                    simulation.nodes(nodesRef.current);
                    simulation.alpha(0.5).restart();
                    if (onNodePositionsChange) {
                        onNodePositionsChange(nodesRef.current.map(node => ({ id: node.id, x: node.x, y: node.y })));
                    }
                    handleTransmitToMainVisualizer();
                }

                const handleKeyDown = (e: KeyboardEvent) => {
                    if (e.key === "x" || e.key === "X") {
                        if (selectedLinkRef.current) {
                            const linkEl = selectedLinkRef.current;
                            const linkDatum = d3.select(linkEl).datum() as any;

                            linksRef.current = linksRef.current.filter(
                                (l) =>
                                    !(
                                        (getEndpointId(l.source) === getEndpointId(linkDatum.source) &&
                                            getEndpointId(l.target) ===
                                                getEndpointId(linkDatum.target)) ||
                                        (getEndpointId(l.source) === getEndpointId(linkDatum.target) &&
                                            getEndpointId(l.target) === getEndpointId(linkDatum.source))
                                    )
                            );

                            selectedLinkRef.current = null;
                        }

                        if (selectedNodeRef.current) {
                            const nodeIdToDelete = selectedNodeRef.current;

                            nodesRef.current = nodesRef.current.filter(
                                (n) => n.id !== nodeIdToDelete
                            );

                            linksRef.current = linksRef.current.filter(
                                (l) =>
                                    getEndpointId(l.source) !== nodeIdToDelete &&
                                    getEndpointId(l.target) !== nodeIdToDelete
                            );

                            clearEditorSelection();
                        }

                        const linkForce = simulationRef.current?.force(
                            "link"
                        ) as d3.ForceLink<any, any>;
                        linkForce?.links(linksRef.current);
                        simulationRef.current?.nodes(nodesRef.current);
                        simulationRef.current?.alpha(0.5).restart();
                        if (onNodePositionsChange) {
                            onNodePositionsChange(nodesRef.current.map(node => ({ id: node.id, x: node.x, y: node.y })));
                        }
                        handleTransmitToMainVisualizer();
                    }
                };

                window.addEventListener("keydown", handleKeyDown);

                return () => {
                    window.removeEventListener("keydown", handleKeyDown);
                    simulation.stop();
                };
    }, [simGraphData]);

    const stopSimulation = () => {
        const sim = simulationRef.current;
        if (!sim) return;

        sim.force("link", null);
        sim.force("charge", null);
        isRunningRef.current = false;
        setIsRunning(false);
        sim.alpha(0.5).restart();
    };

    const startSimulation = () => {
        const sim = simulationRef.current;
        if (!sim) return;

        sim.force(
            "link",
            d3.forceLink(linksRef.current)
                .id((d: any) => d.id)
                .distance(100)
        ).force("charge", d3.forceManyBody().strength(-100));
        isRunningRef.current = true;
        setIsRunning(true);
        sim.alpha(0.5).restart();
    };

    const handleModeSwitch = (nextMode: "node" | "edge") => {
        modeRef.current = nextMode;
        setMode(nextMode);
        clearEditorSelection();

        if (nextMode === "node") {
            stopSimulation();
        } else {
            startSimulation();
        }
    }

    const handleTransmitToMainVisualizer = () => {
        const currentDataset = getCurrentDataset();

        console.log("editor pipe", currentDataset);

        const dataReady = processDataFromEditorToVisualizer(currentDataset);
        if (handleSimulatedGraphChange) {
            handleSimulatedGraphChange(dataReady);
        }
        console.log("Transmitting data to main visualizer:", dataReady);
    };

    return (
            <div className="graph_editor__root">
                <div style={{ padding: "4px" }}>
                    <button
                        onClick={() => handleModeSwitch("node")}
                        style={{
                            padding: "10px 20px",
                            borderRadius: "30px",
                            border: "2px solid #aaa",
                            fontWeight: "bold",
                            backgroundColor: mode === "node" ? "yellow" : "white",
                            cursor: "pointer"
                            , color: "#aaa"
                            }}
                    >
                        Node Edit
                    </button>
                    <button
                        onClick={() => handleModeSwitch("edge")}
                        style={{
                            padding: "10px 20px",
                            borderRadius: "30px",
                            border: "2px solid #aaa",
                            fontWeight: "bold",
                            backgroundColor: mode === "edge" ? "yellow" : "white",
                            cursor: "pointer", color: "#aaa"
                            }}
                    >
                        Edge Edit
                    </button>
                    <div style={{ padding: "8px 4px" }}>
                        <label style={{ fontWeight: "bold", marginRight: "8px", color: "#555" }}>
                            Feature Editor:
                        </label>
                        <input
                            type="text"
                            value={feature}
                            onChange={(e) => setFeature(e.target.value)}
                            placeholder="e.g. 0.1, -0.3, 0.5, 1.2, 0.0"
                            style={{
                                width: "60%",
                                padding: "6px 8px",
                                borderRadius: "4px",
                                border: "1px solid #ccc",
                                fontFamily: "monospace",
                                fontSize: "12px",
                            }}
                        />
                    </div>
                </div>

                <div
                    className="graph_editor__canvas"
                    ref={svgContainer}
                ></div>
            </div>
    );
}
