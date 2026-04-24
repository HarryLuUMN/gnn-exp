import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import type { ResolvedRenderer } from "../renderers/capabilities";
import { interactionPipeline } from "./interactionPipeline";
import { extractSortedGNNLayerFeatures, processSubgraphSequenceDataPipe, transformDataToMatrixVisFormat } from "./utils/dataProcessingUtils";
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
    const adjacancyMatrix = transformDataToMatrixVisFormat(nodeList, linkList);
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);

    const subgraphData = processSubgraphSequenceDataPipe(adjacancyMatrix, queries, 5);
    console.log("subgraphData:", subgraphData);
    
    console.log("Processed nodes and links:", nodeList, linkList);
    
    if (renderer !== "svg") {
        let result: Awaited<ReturnType<typeof staticVisualizationPipeline>>;
        try {
            result = await staticVisualizationPipeline(
                container,
                renderer,
                cellWidth,
                cellHeight,
                adjacancyMatrix,
                intmData,
                linkList,
                queries,
                subgraphSample,
                mode
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

    visualizationPipeline(container, cellWidth, cellHeight, adjacancyMatrix, intmData, linkList, queries, subgraphData, subgraphSample, mode);
    interactionPipeline(container, cellWidth, adjacancyMatrix, sortedGNNFeatures, modelInfo, mode, queries);
    
    return () => {
        container.replaceChildren();
    };
}
