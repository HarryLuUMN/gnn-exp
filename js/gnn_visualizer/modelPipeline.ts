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
    
    // data processing pipes
    const { nodeList, linkList} = preMatrixVisualizationDataProcessingPipe("node prediction", undefined, undefined, graphData);
    const adjacancyMatrix = transformDataToMatrixVisFormat(nodeList, linkList);
    const sortedGNNFeatures = extractSortedGNNLayerFeatures(modelInfo);
    
    console.log("Processed nodes and links:", nodeList, linkList);
    
    // visualization pipes
    visualizationPipeline(adjacancyMatrix, modelInfo, intmData, linkList);
    interactionPipeline(adjacancyMatrix, sortedGNNFeatures);
    
    return null;
}