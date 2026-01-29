import pathlib
import anywidget
import traitlets
from .utils.data_loader import load_json
from .utils.subgraph_sampling import subgraph_hoop_sampling, multiple_subgraph_hoop_sampling

BASE_DIR = pathlib.Path(__file__).resolve().parent

DIST = BASE_DIR / "static"

class GraphVisualizer(anywidget.AnyWidget):
    graphData = traitlets.Dict().tag(sync=True)  

    _esm = DIST / "dual_views" / "index.js"
    _css = DIST / "dual_views" / "index.css"

    value = traitlets.Int(0).tag(sync=True)

    def add_data(self, dataFile):
        self.graphData = load_json(file_path=dataFile)
        print(f"graphData: {self.graphData.keys()}, loaded, path: {DIST}.")

    def subgraph_hoop_visualizer(self, hubNode: int, hoopNum: int):
        if not self.graphData:
            raise ValueError("graphData is empty. Call add_data() first.")

        sub = subgraph_hoop_sampling(self.graphData, hubNode, hoopNum)
        self.graphData = sub 

        print(f"Updated to {hoopNum}-hop subgraph centered at {hubNode}.")

    def multiple_subgraph_hoop_visualizer(self, hubNodes: list, hoopNum: int):
        if not self.graphData:
            raise ValueError("graphData is empty. Call add_data() first.")

        subs = multiple_subgraph_hoop_sampling(self.graphData, hubNodes, hoopNum)
        self.graphData = subs 

        print(f"Updated to {hoopNum}-hop subgraphs centered at {hubNodes}.")

