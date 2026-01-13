import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import { interactionPipeline } from "./interactionPipeline";
import { extractSortedGNNLayerFeatures, processSubgraphDataPipe, transformDataToMatrixVisFormat } from "./utils/dataProcessingUtils";
import { visualizationPipeline } from "./visualizationPipeline";

export function modelPipeline(
    setIsLoading:any, 
    modelInfo:any, 
    intmData:any, 
    graphData:any,
    queries: number[][]
){
    console.log("Starting visualization pipeline...", modelInfo, intmData, graphData, setIsLoading, queries);

    // define parameters
    const cellWidth = 6;
    const cellHeight = 12;

    // data processing pipes
    const { nodeList, linkList} = preMatrixVisualizationDataProcessingPipe("node prediction", undefined, undefined, graphData);
    const adjacancyMatrix = transformDataToMatrixVisFormat(nodeList, linkList);
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(intmData);

    const subgraphData = processSubgraphDataPipe(adjacancyMatrix, queries, 2);
    console.log("subgraphData:", subgraphData);
    
    console.log("Processed nodes and links:", nodeList, linkList);
    
    // visualization pipes
    visualizationPipeline(cellWidth, cellHeight, adjacancyMatrix, intmData, linkList, queries);
    interactionPipeline(cellWidth, adjacancyMatrix, sortedGNNFeatures, modelInfo);
    
    return null;
}

