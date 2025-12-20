import { preMatrixVisualizationDataProcessingPipe } from "../utils/dataProcessingPipeline";
import { interactionPipeline } from "./interactionPipeline";
import { extractSortedGNNLayerFeatures, transformDataToMatrixVisFormat } from "./pipeUtils";
import { visualizationPipeline } from "./visualizationPipeline";

export function modelPipeline(
    setIsLoading:any, 
    modelInfo:any, 
    intmData:any, 
    graphData:any
){
    console.log("Starting visualization pipeline...", modelInfo, intmData, graphData, setIsLoading);

    // define parameters
    const cellWidth = 6;
    const cellHeight = 12;

    // data processing pipes
    const { nodeList, linkList} = preMatrixVisualizationDataProcessingPipe("node prediction", undefined, undefined, graphData);
    const adjacancyMatrix = transformDataToMatrixVisFormat(nodeList, linkList);
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(modelInfo);
    
    console.log("Processed nodes and links:", nodeList, linkList);
    
    // visualization pipes
    visualizationPipeline(cellWidth, cellHeight, adjacancyMatrix, modelInfo, intmData, linkList);
    interactionPipeline(cellWidth, adjacancyMatrix, sortedGNNFeatures);
    
    return null;
}