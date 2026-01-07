import pathlib
import anywidget
import traitlets
import json

ROOT = pathlib.Path(__file__).resolve().parents[2]

DIST = ROOT / "dist"

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
        self.dataFile = browser_url

    def export_data(self):
        return self.graphData
    
    def export_data_to_json(self, output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.graphData, f)
        print(f"Graph data exported to {output_path}")

    @traitlets.observe("graphData")
    def _on_graph_change(self, change):
        print("Python received updated graphData:", change["new"])

