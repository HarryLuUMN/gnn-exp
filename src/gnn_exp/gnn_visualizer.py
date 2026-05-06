import pathlib
import anywidget
import traitlets
import torch
import inspect
import functools
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
    renderer = traitlets.Enum(
        values=["svg", "auto", "webgl", "webgpu"],
        default_value="auto",
    ).tag(sync=True)
    effectiveRenderer = traitlets.Enum(
        values=["svg", "webgl", "webgpu"],
        default_value="webgl",
    ).tag(sync=True)

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
        intermedia_output = self.fetch_model_intermedia(data, model, forward_fn, mode)
        
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
                    "aggregation": self._get_layer_aggregation(module),
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

    def fetch_model_intermedia(self, data, model, forward_fn=None, mode='node'):
        hooks = []
        buffer = {}
        pooling_calls = []

        for name, module in model.named_children():
            h = module.register_forward_hook(
                self.fetch_output_hook(name, buffer)
            )
            hooks.append(h)

        restore_pooling_hooks = self._install_pooling_capture(model, forward_fn, pooling_calls)

        try:
            with torch.no_grad():
                if forward_fn:
                    model_output = forward_fn(model, data)
                else:
                    model_output = self._call_model(model, data, mode)
            self._record_model_output(buffer, model_output)
        finally:
            restore_pooling_hooks()
            for h in hooks:
                h.remove()

        self._record_graph_aggregation(buffer, pooling_calls)

        json_safe_output = {
            name: self.tensor_to_json(value)
            for name, value in buffer.items()
        }

        return json_safe_output

    def fetch_output_hook(self, name, buffer):
        def hook(module, input, output):
            self._record_intermediate_value(buffer, name, output)
        return hook

    @staticmethod
    def _call_model(model, data, mode):
        args = [data.x, data.edge_index]
        batch = GNNVisualizer._get_batch_vector(data)

        try:
            parameters = list(inspect.signature(model.forward).parameters.values())
        except (TypeError, ValueError):
            parameters = []

        positional_names = [
            parameter.name
            for parameter in parameters
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if len(positional_names) == 1 and positional_names[0] == "data":
            return model(data)

        wants_batch = (
            batch is not None
            and (
                "batch" in positional_names
                or (mode == "graph" and len(positional_names) >= 3)
            )
        )
        if wants_batch:
            args.append(batch)

        return model(*args)

    @staticmethod
    def _get_batch_vector(data):
        batch = getattr(data, "batch", None)
        if batch is not None:
            return batch

        x = getattr(data, "x", None)
        if isinstance(x, torch.Tensor):
            return torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        return None

    @staticmethod
    def _record_intermediate_value(buffer, name, value):
        if isinstance(value, torch.Tensor):
            buffer[name] = value.detach()
            return

        if isinstance(value, dict):
            for key, nested in value.items():
                GNNVisualizer._record_intermediate_value(buffer, str(key), nested)
            return

        if isinstance(value, (list, tuple)):
            tensor_values = [
                item.detach()
                for item in value
                if isinstance(item, torch.Tensor)
            ]
            if len(tensor_values) == 1:
                buffer[name] = tensor_values[0]
            elif len(tensor_values) > 1:
                for index, tensor in enumerate(tensor_values):
                    buffer[f"{name}_{index}"] = tensor

    @staticmethod
    def _record_model_output(buffer, output):
        if isinstance(output, dict):
            for key, value in output.items():
                GNNVisualizer._record_intermediate_value(buffer, str(key), value)
        elif isinstance(output, torch.Tensor):
            buffer["modelOutput"] = output.detach()

    @staticmethod
    def _pooling_label(function_name):
        labels = {
            "global_mean_pool": "Mean Pooling",
            "global_add_pool": "Sum Pooling",
            "global_max_pool": "Max Pooling",
        }
        return labels.get(function_name, "Pooling")

    @staticmethod
    def _install_pooling_capture(model, forward_fn, pooling_calls):
        patches = []
        function_names = ("global_mean_pool", "global_add_pool", "global_max_pool")

        def make_wrapper(function_name, original):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                result = original(*args, **kwargs)
                if isinstance(result, torch.Tensor):
                    pooling_calls.append({
                        "name": function_name,
                        "label": GNNVisualizer._pooling_label(function_name),
                        "output": result.detach(),
                    })
                return result

            return wrapper

        def patch_mapping(mapping, function_name):
            if not isinstance(mapping, dict) or function_name not in mapping:
                return
            original = mapping[function_name]
            if not callable(original):
                return
            mapping[function_name] = make_wrapper(function_name, original)
            patches.append(("mapping", mapping, function_name, original))

        def patch_attribute(owner, function_name):
            original = getattr(owner, function_name, None)
            if not callable(original):
                return
            setattr(owner, function_name, make_wrapper(function_name, original))
            patches.append(("attribute", owner, function_name, original))

        globals_to_patch = []
        forward_globals = getattr(getattr(model, "forward", None), "__globals__", None)
        if forward_globals is not None:
            globals_to_patch.append(forward_globals)
        forward_fn_globals = getattr(forward_fn, "__globals__", None)
        if forward_fn_globals is not None:
            globals_to_patch.append(forward_fn_globals)

        for function_name in function_names:
            for mapping in globals_to_patch:
                patch_mapping(mapping, function_name)

        try:
            import torch_geometric.nn as pyg_nn
            import torch_geometric.nn.pool as pyg_pool
            import torch_geometric.nn.pool.glob as pyg_pool_glob
        except Exception:
            pyg_nn = pyg_pool = pyg_pool_glob = None

        for owner in (pyg_nn, pyg_pool, pyg_pool_glob):
            if owner is None:
                continue
            for function_name in function_names:
                patch_attribute(owner, function_name)

        def restore():
            for patch_type, target, function_name, original in reversed(patches):
                if patch_type == "mapping":
                    target[function_name] = original
                else:
                    setattr(target, function_name, original)

        return restore

    @staticmethod
    def _normalize_graph_feature(value):
        if not isinstance(value, torch.Tensor):
            return None

        feature = value.detach()
        if feature.ndim == 0:
            return None
        if feature.ndim == 1:
            return feature
        if feature.ndim >= 2 and feature.shape[0] > 0:
            return feature.reshape(feature.shape[0], -1)[0]
        return None

    @staticmethod
    def _record_graph_aggregation(buffer, pooling_calls):
        if pooling_calls:
            latest = pooling_calls[-1]
            feature = GNNVisualizer._normalize_graph_feature(latest["output"])
            if feature is not None:
                buffer["graphAggregation"] = {
                    "type": latest["label"],
                    "name": latest["name"],
                    "feature": feature,
                    "features": latest["output"],
                }
                return

        for key, value in buffer.items():
            normalized_key = key.lower().replace("_", "").replace("-", "")
            if not any(token in normalized_key for token in ("pool", "readout", "graphembedding", "graphfeature", "graphrepr")):
                continue
            feature = GNNVisualizer._normalize_graph_feature(value)
            if feature is None:
                continue
            buffer["graphAggregation"] = {
                "type": "Pooling",
                "name": key,
                "feature": feature,
                "features": value,
            }
            return

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
    def _normalize_aggregation_name(value):
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            return [GNNVisualizer._normalize_aggregation_name(item) for item in value]

        raw = value if isinstance(value, str) else type(value).__name__
        normalized = raw.strip().lower().replace("_", "-").replace(" ", "")
        if normalized.endswith("()"):
            normalized = normalized[:-2]
        if normalized.endswith("aggregation"):
            normalized = normalized[:-len("aggregation")]

        aliases = {
            "gcn": "gcn-normalized",
            "gcnconv": "gcn-normalized",
            "gcn-normalized": "gcn-normalized",
            "gcn-normalised": "gcn-normalized",
            "normalized-gcn": "gcn-normalized",
            "normalised-gcn": "gcn-normalized",
            "add": "sum",
            "sum": "sum",
            "avg": "mean",
            "average": "mean",
            "mean": "mean",
            "maximum": "max",
            "max": "max",
            "minimum": "min",
            "min": "min",
            "median": "median",
            "stdev": "std",
            "std": "std",
            "variance": "var",
            "var": "var",
        }
        return aliases.get(normalized, raw)

    @staticmethod
    def _get_layer_aggregation(module):
        module_type = type(module).__name__
        if module_type == "GCNConv":
            return "gcn-normalized"

        aggregation = getattr(module, "aggr", None)
        return GNNVisualizer._normalize_aggregation_name(aggregation) or "gcn-normalized"

    @staticmethod
    def tensor_to_json(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        if isinstance(x, dict):
            return {key: GNNVisualizer.tensor_to_json(value) for key, value in x.items()}
        if isinstance(x, (list, tuple)):
            return [GNNVisualizer.tensor_to_json(value) for value in x]
        return x
