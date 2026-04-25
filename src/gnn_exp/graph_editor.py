import pathlib
import anywidget
import traitlets
import json

BASE_DIR = pathlib.Path(__file__).resolve().parent

DIST = BASE_DIR / "static"

class GraphEditor(anywidget.AnyWidget):
    dataFile = traitlets.Unicode("/files/test_data/karate_dataset.json").tag(sync=True)
    graphData = traitlets.Dict().tag(sync=True)

    _esm = DIST / "graph_editor" / "index.js"
    _css = DIST / "graph_editor" / "index.css"

    value = traitlets.Int(0).tag(sync=True)

    def add_data(self, dataFile: str):
        file_path = dataFile.lstrip("/")  
        browser_url = f"/files/{file_path}"
        print("Exposing file to browser:", browser_url)
        with open(file_path, "r") as f:
            self.graphData = json.load(f)
        self.dataFile = browser_url

    def export_data(self):
        return self.graphData
    
    def export_data_to_json(self, output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.graphData, f)
        print(f"Graph data exported to {output_path}")

    @traitlets.observe("graphData")
    def _on_graph_change(self, change):
        graph = change["new"] or {}
        node_count = len(graph.get("x", []))
        edge_index = graph.get("edge_index", [[], []])
        edge_count = len(edge_index[0]) if edge_index else 0
        print(f"Python received updated graphData: {node_count} nodes, {edge_count} edges")
