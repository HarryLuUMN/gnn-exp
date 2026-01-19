import pathlib
import anywidget
import traitlets
import torch
from .utils.data_loader import load_json
from .utils.subgraph_sampling import subgraph_hoop_sampling, multiple_subgraph_hoop_sampling

BASE_DIR = pathlib.Path(__file__).resolve().parent

DIST = BASE_DIR / "static"

class GNNVisualizer(anywidget.AnyWidget):
    graphData = traitlets.Dict().tag(sync=True)  
    graphPath = traitlets.Unicode("").tag(sync=True)
    modelInfo = traitlets.Dict().tag(sync=True)
    intmData = traitlets.Dict().tag(sync=True)
    subgraphSample = traitlets.Bool(False).tag(sync=True)
    mode = traitlets.Unicode("").tag(sync=True)

    queries = traitlets.List(default_value=[]).tag(sync=True)

    renderToken = traitlets.Int(0).tag(sync=True)

    _esm = DIST / "gnn_visualizer" / "index.js"
    _css = DIST / "gnn_visualizer" / "index.css"

    value = traitlets.Int(0).tag(sync=True)

    @traitlets.observe("queries")
    def _on_queries_change(self, change):
        print(f"Python: queries changed to: {change['new']}, type: {type(change['new'])}")

    def add_data(self, graphFile, weightFile, modelInfo, subgraphSample, mode):
        self.graphData = load_json(file_path=graphFile)
        self.intmData = load_json(file_path=weightFile)
        self.modelInfo = modelInfo
        self.subgraphSample = subgraphSample
        self.mode = mode
        print(f"graphData: {self.graphData.keys()}, intmData: {self.intmData.keys()} loaded, path: {DIST}.")

    def add_model(self, data, model, subgraphSample, forward_fn=None, queries=[], mode='node'):
        self.subgraphSample = subgraphSample    
        self.mode = mode
        print(f"mode: {self.mode}")
        intermedia_output = self.fetch_model_intermedia(data, model, forward_fn)
        
        # deduplicate while preserving order and ensure queries are JSON-serializable
        seen = []
        queries_list = []
        for q in queries:
            # Convert to tuple for hashing, then back to list
            q_tuple = tuple(q) if isinstance(q, (list, tuple)) else (q,)
            if q_tuple not in seen:
                seen.append(q_tuple)
                # Ensure each query is a list of integers
                queries_list.append([int(x) for x in q] if isinstance(q, (list, tuple)) else [int(q)])
        
        self.queries = queries_list
        print(f"Queries set to: {self.queries}")
        self.intmData = intermedia_output
        self.graphData = {
            "x": data.x.detach().cpu().numpy().tolist(),
            "edge_index": data.edge_index.detach().cpu().numpy().tolist(),
            "y": data.y.detach().cpu().numpy().tolist() if data.y is not None else None,
        }

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
            elif self._is_activation_module(module):
                activation_type = self._get_activation_type(module)
                model_info[name] = {
                    "type": activation_type,
                }

        self.modelInfo = model_info
        self.intmData = {**self.intmData, 'act0': data.x.detach().cpu().numpy().tolist()}

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

    def fetch_model_intermedia(self, data, model, forward_fn=None):
        hooks = []
        buffer = {}

        for name, module in model.named_children():
            h = module.register_forward_hook(
                self.fetch_output_hook(name, buffer)
            )
            hooks.append(h)

        with torch.no_grad():
            if forward_fn:
                _ = forward_fn(model, data)
            else:
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
    def _is_activation_module(module):
        """Check if a module is an activation function."""
        activation_types = [
            torch.nn.ReLU,
            torch.nn.Tanh,
            torch.nn.Sigmoid,
            torch.nn.Softmax,
            torch.nn.LeakyReLU,
            torch.nn.ELU,
            torch.nn.GELU,
            torch.nn.ReLU6,
            torch.nn.SELU,
            torch.nn.Softplus,
        ]
        # Check for Swish (may not exist in all PyTorch versions)
        try:
            activation_types.append(torch.nn.Swish)
        except AttributeError:
            pass
        
        return isinstance(module, tuple(activation_types))

    @staticmethod
    def _get_activation_type(module):
        """Get the string name of the activation function type."""
        activation_map = {
            torch.nn.ReLU: "ReLU",
            torch.nn.Tanh: "Tanh",
            torch.nn.Sigmoid: "Sigmoid",
            torch.nn.Softmax: "Softmax",
            torch.nn.LeakyReLU: "LeakyReLU",
            torch.nn.ELU: "ELU",
            torch.nn.GELU: "GELU",
            torch.nn.ReLU6: "ReLU6",
            torch.nn.SELU: "SELU",
            torch.nn.Softplus: "Softplus",
        }
        # Handle Swish which might be defined differently or not exist
        module_type = type(module)
        if module_type.__name__ == "Swish":
            return "Swish"
        return activation_map.get(module_type, module_type.__name__)

    @staticmethod
    def tensor_to_json(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        return x
