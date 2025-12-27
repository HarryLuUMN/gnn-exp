import { NodeDatum, LinkDatum } from "../dual_views/dualViewTypes";

const elementMap: Record<number, string> = {
    0: "C",
    1: "N",
    2: "O",
    3: "F",
    4: "H",
    5: "S",
    6: "Cl",
};


export function preMatrixVisualizationDataProcessingPipe(
    modelType: any,
    hubNodeA: number | undefined,
    hubNodeB: number | undefined,
    data: any,
    sandboxMode: boolean = false
) {
    console.log("preMatrixVisualizationDataProcessingPipe:", modelType, hubNodeA, hubNodeB, data);
    let processed: number[] = [];
    if (modelType?.includes("link prediction")) {
        const sub = new Set<number>();
        if (hubNodeA != null) sub.add(hubNodeA);
        if (hubNodeB != null) sub.add(hubNodeB);
        if (data.edge_index) {
            for (let i = 0; i < data.edge_index[0].length; i++) {
                const s = data.edge_index[0][i];
                const t = data.edge_index[1][i];
                if (s === hubNodeA || s === hubNodeB) sub.add(t);
                if (t === hubNodeA || t === hubNodeB) sub.add(s);
            }
        }
        processed = Array.from(sub).sort((a, b) => a - b);
    } else {
        processed = data.x.map((_v: any, i: number) => i);
    }

    const nodeList: NodeDatum[] = processed.map((nodeId: number) => {
        let label = String(nodeId);
        if (!sandboxMode && modelType?.includes("node prediction")) {
            const feats = data.x[nodeId];
            const idx = Array.isArray(feats) ? feats.indexOf(1) : -1;
            if (idx !== -1 && elementMap[idx] != null) label = elementMap[idx];
            else if (data.train_nodes) label = data.train_nodes.includes(nodeId) ? "T" : "?";
            else if (data.y) label = String(data.y[nodeId]);
        }
        return { id: nodeId, element: label };
    });

    // filter links to processed nodes subset
    const setProcessed = new Set(processed);
    const linkList: LinkDatum[] = (data.edge_index?.[0] || []).reduce(
        (acc: LinkDatum[], s: number, i: number) => {
            const t = data.edge_index[1][i];
            if (setProcessed.has(s) && setProcessed.has(t)) {
                acc.push({ source: s, target: t, attr: data.edge_attr ? data.edge_attr[i] : null });
            }
            return acc;
        },
        []
    );

    return { nodeList, linkList };
}
