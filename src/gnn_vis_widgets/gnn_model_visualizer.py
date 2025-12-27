import pathlib
import anywidget
import traitlets
import torch
from .utils.data_loader import load_json
from .utils.subgraph_sampling import subgraph_hoop_sampling, multiple_subgraph_hoop_sampling

ROOT = pathlib.Path(__file__).resolve().parents[2]

DIST = ROOT / "dist"

class GNNVisualizer(anywidget.AnyWidget):
    graphData = traitlets.Dict().tag(sync=True)  
    graphPath = traitlets.Unicode("").tag(sync=True)
    modelInfo = traitlets.Dict().tag(sync=True)
    intmData = traitlets.Dict().tag(sync=True)

    renderToken = traitlets.Int(0).tag(sync=True)

    _esm = DIST / "gnn_visualizer" / "index.js"
    _css = DIST / "gnn_visualizer" / "index.css"

    value = traitlets.Int(0).tag(sync=True)

    def add_data(self, graphFile, weightFile, modelInfo):
        self.graphData = load_json(self=self,file_path=graphFile, root=ROOT)
        self.intmData = load_json(self=self,file_path=weightFile, root=ROOT)
        self.modelInfo = modelInfo
        print(f"graphData: {self.graphData.keys()}, intmData: {self.intmData.keys()} loaded.")

    def add_model(self, data, model):
        intermedia_output = self.fetch_model_intermedia(data, model)
        self.intmData = intermedia_output

        model_info = {}

        for name, module in model.named_modules():
            if hasattr(module, "lin"):  # GCNConv
                model_info[name] = {
                    "type": "GCNConv",
                    "weight": self.tensor_to_json(module.lin.weight),
                    "bias": self.tensor_to_json(module.bias),
                }
            elif isinstance(module, torch.nn.Linear):
                model_info[name] = {
                    "type": "Linear",
                    "weight": self.tensor_to_json(module.weight),
                    "bias": self.tensor_to_json(module.bias),
                }

        self.modelInfo = model_info
        self.intmData = {**self.intmData, 'act0': data['x'].detach().cpu().numpy().tolist()}

        print(f"check act0: {self.intmData['act0'][:5]}")

        print(
            f"modelInfo: {self.modelInfo.keys()}, "
            f"intmData: {self.intmData.keys()} loaded."
        )

        if self.modelInfo:
            last_layer_name = list(self.modelInfo.keys())[-1]
            last_layer_info = self.modelInfo[last_layer_name]
            if "bias" in last_layer_info:
                print(f"Last layer ({last_layer_name}) bias: {last_layer_info['bias']}")
            else:
                print(f"Last layer ({last_layer_name}) has no bias")

        self.renderToken += 1

    def start_visualize(self):
        self.value += 1  # trigger re-render in frontend
        return 

    def fetch_model_intermedia(self, data, model):
        hooks = []
        buffer = {}

        for name, module in model.named_children():
            h = module.register_forward_hook(
                self.fetch_output_hook(name, buffer)
            )
            hooks.append(h)

        with torch.no_grad():
            _ = model(data.x, data.edge_index)

        for h in hooks:
            h.remove()

        json_safe_output = {
            name: self.tensor_to_json(tensor)
            for name, tensor in buffer.items()
        }

        return json_safe_output

    def fetch_output_hook(self, name, buffer):
        def hook(module, input, output):
            buffer[name] = output.detach()
        return hook

    @staticmethod
    def tensor_to_json(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        return x
