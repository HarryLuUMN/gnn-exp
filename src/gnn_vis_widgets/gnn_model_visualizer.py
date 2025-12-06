import pathlib
import anywidget
import traitlets
from .utils.data_loader import load_json
from .utils.subgraph_sampling import subgraph_hoop_sampling, multiple_subgraph_hoop_sampling

ROOT = pathlib.Path(__file__).resolve().parents[2]

DIST = ROOT / "dist"

class GNNVisualizer(anywidget.AnyWidget):
    graphData = traitlets.Dict().tag(sync=True)  
    graphPath = traitlets.Unicode("").tag(sync=True)
    intmData = traitlets.Dict().tag(sync=True)

    _esm = DIST / "gnn_visualizer" / "index.js"
    _css = DIST / "gnn_visualizer" / "index.css"

    value = traitlets.Int(0).tag(sync=True)

    def add_data(self, graphFile, weightFile):
        self.graphData = load_json(self=self,file_path=graphFile, root=ROOT)
        self.intmData = load_json(self=self,file_path=weightFile, root=ROOT)
        print(f"graphData: {self.graphData.keys()}, intmData: {self.intmData.keys()} loaded.")

