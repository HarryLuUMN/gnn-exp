# TODO: implement the feature data processing pipeline first. 

import numpy as np

# model_info: information about the GNN model architecture and parameters
# - model_architecture: str [] -> gnn_layer_{x}, pooling_layer_{x}, fc_layer_{x}
# - layer_dimensions: dict -> {layer_name: dimension -> str: {dim1}-{dim2}}

# the pipeline processes the weights data for GNN model visualization
# ordering -> transforming -> modeling
def weights_data_processing_pipe(model_info: dict, intermadiate_data: dict):
    pass

# the pipeline processes the bias data for GNN model visualization
# ordering -> transforming -> modeling
def bias_data_processing_pipe(model_info: dict, intermadiate_data: dict):
    pass

# the pipeline processes the intermediate feature data for GNN model visualization
# ordering -> transforming -> modeling
def intermdiate_data_partitioning_pipe(intermadiate_data: dict):
    pass

# the pipeline processes the whole data for GNN model visualization
# ordering -> transforming -> modeling
def model_data_processing_pipe(model_info: dict, intermadiate_data: dict):
    pass

# the pipeline models the entire graph data for GNN model visualization
def graph_data_modeling_pipe(model_info: dict, intermadiate_data: dict, graph_data: dict):
    pass

# the pipeline models the subgraph data for GNN model visualization
def subgraph_data_modeling_pipe(model_info: dict, intermadiate_data: dict, graph_data: dict):
    pass