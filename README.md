# GNN Explorer: Visual Exploration and Explanation of Graph Neural Networks

A comprehensive visualization library for Graph Neural Networks (GNNs) that enables interactive exploration and explanation of GNN models directly in computational notebooks. This library provides multiple visualization widgets for understanding graph structures, model architectures, and intermediate feature representations.

## Python Package Index

We have published the public version to Python Package Index, use

```sh
pip install gnn-exp
```

to install the package. 

## Release Publishing

This repository is configured for PyPI Trusted Publishing with GitHub Actions. Publishing is handled by [`.github/workflows/publish-pypi.yml`](/Users/harrylu_mac/gnn-explorer/.github/workflows/publish-pypi.yml), which builds the frontend assets, creates the Python distributions, and uploads them to PyPI through GitHub OIDC.

To finish the one-time PyPI setup, add a Trusted Publisher for:

- PyPI project: `gnn-exp`
- GitHub owner: `HarryLuUMN`
- GitHub repository: `gnn-exp`
- Workflow name: `Publish Python Package`
- Environment name: `pypi`

After that, publishing can be triggered either by creating a GitHub Release or by manually running the workflow from the Actions tab.

## Features

### 🎨 **GraphVisualizer** - Dual View Graph Visualization
- Visualize graphs with both node-link and matrix views
- Support for large graphs with subgraph extraction
- Hoop-based subgraph sampling for focused visualization
- Interactive exploration of graph structures

### ✏️ **GraphEditor** - Interactive Graph Editing
- Edit graph structures directly in the notebook
- Add, remove, and modify nodes and edges
- Export edited graphs to JSON format
- Real-time visualization updates

### 🧠 **GNNVisualizer** - Model Feature Visualization
- Visualize intermediate layer features and activations
- Matrix view of feature transformations across layers
- Support for multiple GNN tasks:
  - **Node Classification**: Visualize node-level predictions
  - **Edge Prediction**: Visualize link prediction tasks
  - **Graph Classification**: Visualize graph-level tasks
- Interactive layer expansion to explore inner computations
- Subgraph-based feature visualization for large graphs

## Installation

### From Source

Clone the repository and install in development mode:

```sh
git clone https://github.com/HarryLuUMN/gnn-exp.git
cd gnn-exp
```

### Python Dependencies

We recommend using [uv](https://github.com/astral-sh/uv) for development, which automatically manages virtual environments and dependencies:

```sh
# Install Python dependencies
uv sync

# Run example notebook
uv run jupyter lab example.ipynb
```

Alternatively, use a traditional virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
jupyter lab example.ipynb
```

### JavaScript Dependencies

The widget front-end code bundles JavaScript dependencies. After setting up Python, install JavaScript dependencies:

```sh
npm install
```

For development with hot-reloading, run in a separate terminal:

```sh
npm run dev
```

This will automatically rebuild JavaScript as you make changes to files in `js/`.

## Quick Start

### Visualizing Graphs

```python
from gnn_explorer import GraphVisualizer

# Create a visualizer instance
w = GraphVisualizer()
w.add_data(dataFile="test_data/karate_dataset.json")
w  # Display in notebook
```

By default the widgets use `renderer="auto"`: WebGL is the baseline renderer, and browsers with a usable WebGPU adapter automatically promote to WebGPU. To force a backend from Python, set `renderer` to `"webgl"`, `"webgpu"`, or `"svg"`:

```python
w = GraphVisualizer(renderer="webgl")
# GNNVisualizer accepts the same renderer keyword.
```

### Visualizing Large Graphs with Subgraphs

```python
# Single hub node subgraph extraction
w = GraphVisualizer()
w.add_data(dataFile="test_data/twitch.json")
w.subgraph_hoop_visualizer(hubNode=0, hoopNum=3)
w

# Multiple hub nodes
w = GraphVisualizer()
w.add_data(dataFile="test_data/twitch.json")
w.multiple_subgraph_hoop_visualizer(hubNodes=[0, 1], hoopNum=1)
w
```

### Editing Graphs

```python
from gnn_explorer import GraphEditor

editor = GraphEditor()
editor.add_data(dataFile="test_data/karate_dataset.json")
editor  # Display editor in notebook

# Export edited graph
editor.export_data_to_json("test_data/edited_graph.json")
```

### Visualizing GNN Models

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from gnn_explorer import GNNVisualizer

# Define your GNN model
class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.act1 = nn.Tanh()
        self.conv2 = GCNConv(4, 4)
        self.act2 = nn.Tanh()
        self.classifier = nn.Linear(4, dataset.num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, edge_index):
        h = self.act1(self.conv1(x, edge_index))
        h = self.act2(self.conv2(h, edge_index))
        out = self.softmax(self.classifier(h))
        return out

# Create model and data
model = GCN(dataset)
data = dataset[0]

# Visualize model
visualizer = GNNVisualizer()
visualizer.add_model(
    data=data,
    model=model,
    subgraphSample=False,
    forward_fn=None,
    queries=[[12, 18]],  # Node IDs to highlight
    mode='node'  # 'node', 'edge', or 'graph'
)
visualizer  # Display in notebook
```

## Project Structure

```
gnn-explorer/
├── src/gnn_explorer/          # Python package
│   ├── graph_visualizer.py    # Graph visualization widget
│   ├── graph_editor.py        # Graph editing widget
│   ├── gnn_visualizer.py      # GNN model visualization widget
│   └── static/                # Compiled JavaScript/CSS assets
├── js/                         # TypeScript/React source code
│   ├── graph_visualizer/      # Graph visualization frontend
│   ├── graph_editor/          # Graph editor frontend
│   └── gnn_visualizer/        # GNN visualizer frontend
├── test_data/                 # Example datasets
├── example.ipynb              # Example notebook
└── README.md                  # This file
```

## Development

### Building JavaScript

The JavaScript code is built using `esbuild`. To build once:

```sh
npm run build
```

To watch for changes during development:

```sh
npm run dev
```

### Type Checking

Run TypeScript type checking:

```sh
npm run typecheck
```

### Hot Module Reloading

For development, enable hot module reloading in your notebook:

```python
%load_ext autoreload
%autoreload 2
%env ANYWIDGET_HMR=1
```

Changes made to files in `js/` will be automatically reflected in the notebook when using `npm run dev`.

## Requirements

- Python >= 3.8
- Node.js >= 16 (for development)
- JupyterLab or Jupyter Notebook
- PyTorch Geometric (for GNN model visualization)

## Supported GNN Layers

- **GCNConv**: Graph Convolutional Network layers
- **GATConv**: Graph Attention Network layers with basic rendering support
- **SAGEConv / GraphSAGEConv**: GraphSAGE layers with supported aggregation labels
- **GINConv**: Graph Isomorphism Network layers with sum aggregation and first linear projection rendering
- **Linear**: Fully connected layers
- **Activation Functions**: ReLU, Tanh, Sigmoid, Softmax, and more

GATConv support currently renders the layer structure and learned projection weights, but does not display exact per-edge or per-head attention coefficients.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Authors

- Yilin Lu (lu000661@umn.edu)
- Qianwen Wang (qianwen@umn.edu)

## Repository

- **Homepage**: https://github.com/HarryLuUMN/gnn-exp
- **Repository**: https://github.com/HarryLuUMN/gnn-exp

## Citation

If you use this library in your research, please cite:

```bibtex
@software{gnn_explorer,
  title = {GNN Explorer: Visual Exploration and Explanation of Graph Neural Networks},
  author = {Lu, Yilin and Wang, Qianwen},
  url = {https://pypi.org/project/gnn-exp/0.1.0/},
  journal = {Python Package Index}
  year = {2024}
}
```
