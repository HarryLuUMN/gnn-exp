import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import { interactionPipeline } from "./interactionPipeline";
import { extractSortedGNNLayerFeatures, processSubgraphSequenceDataPipe, transformDataToMatrixVisFormat } from "./utils/dataProcessingUtils";
import { visualizationPipeline } from "./visualizationPipeline";

export function modelPipeline(
    container: HTMLDivElement,
    setIsLoading:any, 
    modelInfo:any, 
    intmData:any, 
    graphData:any,
    queries: number[][],
    subgraphSample: any,
    mode: string
){
    console.log("Starting visualization pipeline...", modelInfo, intmData, graphData, setIsLoading, queries, mode);

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
    
    // visualization pipes
    visualizationPipeline(container, cellWidth, cellHeight, adjacancyMatrix, intmData, linkList, queries, subgraphData, subgraphSample, mode);
    interactionPipeline(container, cellWidth, adjacancyMatrix, sortedGNNFeatures, modelInfo, mode, queries);
    
    return null;
}
