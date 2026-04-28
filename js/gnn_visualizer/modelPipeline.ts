import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import type { ResolvedRenderer } from "../renderers/capabilities";
import { interactionPipeline } from "./interactionPipeline";
import {
    buildModelVisualizationData,
    extractSortedGNNLayerFeatures,
    processSubgraphSequenceDataPipe,
    transformDataToMatrixVisFormat,
} from "./utils/dataProcessingUtils";
import { staticVisualizationPipeline } from "./staticVisualizationPipeline";
import { visualizationPipeline } from "./visualizationPipeline";

export type PipelineCleanup = () => void;
export type RendererFailureHandler = (
    renderer: Exclude<ResolvedRenderer, "svg">,
    reason: string
) => void;

export async function modelPipeline(
    container: HTMLDivElement,
    setIsLoading:any, 
    modelInfo:any, 
    intmData:any, 
    graphData:any,
    queries: number[][],
    subgraphSample: any,
    mode: string,
    renderer: ResolvedRenderer = "svg",
    onRendererFailure?: RendererFailureHandler
): Promise<PipelineCleanup | null> {
    console.log("Starting visualization pipeline...", modelInfo, intmData, graphData, setIsLoading, queries, mode, renderer);

    // define parameters
    const cellWidth = 6;
    const cellHeight = 12;

    // data processing pipes
    const { nodeList, linkList} = preMatrixVisualizationDataProcessingPipe("node prediction", undefined, undefined, graphData);
    const fullAdjacencyMatrix = transformDataToMatrixVisFormat(nodeList, linkList);
    const viewData = buildModelVisualizationData(
        fullAdjacencyMatrix,
        linkList,
        intmData,
        modelInfo,
        queries,
        mode
    );
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(viewData.intmData);

    const subgraphData = processSubgraphSequenceDataPipe(
        viewData.adjacencyMatrix,
        viewData.queries,
        viewData.messagePassingDepth
    );
    console.log("subgraphData:", subgraphData);
    
    console.log("Processed nodes and links:", nodeList, linkList);
    console.log("Model visualization view data:", viewData);

    const effectiveSubgraphSample = viewData.isLocalEdgeView ? false : subgraphSample;
    
    if (renderer !== "svg") {
        let result: Awaited<ReturnType<typeof staticVisualizationPipeline>>;
        try {
            result = await staticVisualizationPipeline(
                container,
                renderer,
                cellWidth,
                cellHeight,
                viewData.adjacencyMatrix,
                viewData.intmData,
                modelInfo,
                viewData.linkList,
                viewData.queries,
                effectiveSubgraphSample,
                mode,
                viewData.nodeLabels,
                viewData.messagePassingDepth
            );
        } catch (error) {
            const reason =
                error instanceof Error ? error.message : "Static GPU renderer failed.";
            onRendererFailure?.(renderer, reason);
            return null;
        }

        if ("error" in result) {
            onRendererFailure?.(renderer, result.error);
            return null;
        }

        return result.cleanup;
    }

    visualizationPipeline(
        container,
        cellWidth,
        cellHeight,
        viewData.adjacencyMatrix,
        viewData.intmData,
        viewData.linkList,
        viewData.queries,
        subgraphData,
        effectiveSubgraphSample,
        mode,
        viewData.nodeLabels,
        viewData.messagePassingDepth
    );
    interactionPipeline(
        container,
        cellWidth,
        viewData.adjacencyMatrix,
        sortedGNNFeatures,
        modelInfo,
        mode,
        viewData.queries
    );
    
    return () => {
        container.replaceChildren();
    };
}
