import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import type { LinkDatum, NodeDatum } from "../dual_views/dualViewTypes";
import {
    getLinkStrokeColor,
    getNodeFillColor,
    getNodeStrokeColor,
} from "../dual_views/renderers/shared";
import { layoutGraph } from "../dual_views/useGraphScene";
import {
    loadSimGraphData,
    parseFeatureText,
    processDataFromEditorToVisualizer,
    processDataFromVisualizerToEditor,
    randomizeFeatures,
} from "./graphEditorUtils";

const GRAPH_EDITOR_CANVAS_SIZE = 640;
const GRAPH_EDITOR_LAYOUT_PADDING = 60;

function numericNodeId(raw: unknown, fallback: number) {
    const value =
        typeof raw === "object" && raw !== null && "id" in raw
            ? (raw as { id?: unknown }).id
            : raw;

    if (typeof value === "number" && Number.isFinite(value)) {
        return value;
    }

    if (typeof value === "string") {
        const match = value.match(/\d+/);
        if (match) {
            return Number(match[0]);
        }
    }

    return fallback;
}

function rawEndpointId(endpoint: unknown) {
    if (typeof endpoint === "object" && endpoint !== null && "id" in endpoint) {
        return (endpoint as { id?: unknown }).id;
    }

    return endpoint;
}

function editorNodeToGraphDatum(node: any, index: number): NodeDatum {
    const id = numericNodeId(node.id, index);
    return {
        id,
        element: String(id),
        community: node.community,
    };
}

function editorLinkToGraphDatum(link: any, index: number): LinkDatum {
    return {
        source: numericNodeId(rawEndpointId(link.source), index),
        target: numericNodeId(rawEndpointId(link.target), index),
        attr: link.attr ?? (link.value == null ? undefined : { value: link.value }),
    };
}

function getEditorNodeFill(node: any) {
    return getNodeFillColor(null, editorNodeToGraphDatum(node, 0));
}

function getEditorNodeStroke(node: any) {
    return getNodeStrokeColor(null, editorNodeToGraphDatum(node, 0));
}

function getEditorLinkStroke(link: any) {
    return getLinkStrokeColor(null, editorLinkToGraphDatum(link, 0));
}

function pinEditorNodePosition(node: any) {
    if (typeof node.x !== "number" || typeof node.y !== "number") {
        return;
    }

    node.fx = node.x;
    node.fy = node.y;
}

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
                .attr("stroke", (d: any) => getEditorNodeStroke(d));
            d3.select(svgContainer.current)
                .selectAll("line")
                .attr("stroke", (d: any) => getEditorLinkStroke(d));
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

        const width = GRAPH_EDITOR_CANVAS_SIZE;
        const height = GRAPH_EDITOR_CANVAS_SIZE;
        let data = simGraphData;
                console.log("Loaded graph:", data);

                const initialData = processDataFromVisualizerToEditor(data);
                datasetRef.current = initialData;

                const sharedLayoutNodes = initialData.nodes.map(
                    (node: any, index: number) => editorNodeToGraphDatum(node, index)
                );
                const sharedLayoutLinks = initialData.links.map(
                    (link: any, index: number) => editorLinkToGraphDatum(link, index)
                );
                const laidOutNodes = layoutGraph(
                    sharedLayoutNodes,
                    sharedLayoutLinks,
                    width,
                    height,
                    GRAPH_EDITOR_LAYOUT_PADDING
                );
                const layoutByNodeId = new Map(
                    laidOutNodes.map((node) => [node.id, node])
                );

                const links = initialData.links.map((d: any) => ({ ...d }));
                const nodes = initialData.nodes.map((d: any, index: number) => {
                    const graphNode = editorNodeToGraphDatum(d, index);
                    const laidOutNode = layoutByNodeId.get(graphNode.id);
                    const x = laidOutNode?.x ?? width / 2;
                    const y = laidOutNode?.y ?? height / 2;

                    return {
                        ...d,
                        community: laidOutNode?.community,
                        x,
                        y,
                        fx: x,
                        fy: y,
                    };
                });

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
                                    (linkDatum: any) => getEditorLinkStroke(linkDatum)
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
                        .attr("stroke", function (this: d3.BaseType, d: any) {
                            return selectedLinkRef.current === this
                                ? "black"
                                : getEditorLinkStroke(d);
                        })
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
                                    (linkDatum: any) => getEditorLinkStroke(linkDatum)
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
                        .attr("stroke", (d: any) =>
                            selectedNodeRef.current === d.id ||
                            secondSelectedNodeRef.current === d.id
                                ? "black"
                                : getEditorNodeStroke(d)
                        )
                        .attr("fill", (d: any) => getEditorNodeFill(d))
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
                                d3.select(this).attr("stroke", getEditorNodeStroke(d));
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
                        pinEditorNodePosition(event.subject);
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
                ticked();

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
        nodesRef.current.forEach(pinEditorNodePosition);
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
