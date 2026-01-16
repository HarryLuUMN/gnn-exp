import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import { interactionPipeline } from "./interactionPipeline";
import { extractSortedGNNLayerFeatures, processSubgraphSequenceDataPipe, transformDataToMatrixVisFormat } from "./utils/dataProcessingUtils";
import { visualizationPipeline } from "./visualizationPipeline";
import { initCanvasId, initSvgId } from "../states";

export function modelPipeline(
    setIsLoading:any, 
    modelInfo:any, 
    intmData:any, 
    graphData:any,
    queries: number[][],
    subgraphSample: any
){
    console.log("Starting visualization pipeline...", modelInfo, intmData, graphData, setIsLoading, queries);

    initCanvasId();
    initSvgId();

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
    visualizationPipeline(cellWidth, cellHeight, adjacancyMatrix, intmData, linkList, queries, subgraphData, subgraphSample);
    interactionPipeline(cellWidth, adjacancyMatrix, sortedGNNFeatures, modelInfo);
    
    return null;
}

