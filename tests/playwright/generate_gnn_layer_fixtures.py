import json
from pathlib import Path

import torch
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from gnn_exp import GNNVisualizer


class Data:
    def __init__(self):
        self.x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 1],
                [1, 2, 3, 0, 3],
            ]
        )
        self.y = torch.tensor([0, 1, 0, 1])


class Model(torch.nn.Module):
    def __init__(self, conv, with_softmax=True):
        super().__init__()
        self.conv1 = conv
        hidden_dim = 4 if type(conv).__name__ == "GATConv" else 2
        self.act1 = torch.nn.Tanh()
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        h = self.act1(self.conv1(x, edge_index))
        logits = self.classifier(h)
        return self.softmax(logits) if self.with_softmax else logits


def initialize_linear(linear, offset):
    with torch.no_grad():
        values = torch.arange(linear.weight.numel(), dtype=torch.float32)
        linear.weight.copy_(values.reshape_as(linear.weight) * 0.03 + offset)
        if linear.bias is not None:
            bias = torch.arange(linear.bias.numel(), dtype=torch.float32)
            linear.bias.copy_(bias * 0.02 - offset)


def initialize_model(model):
    for index, module in enumerate(model.modules()):
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            initialize_linear(module, offset=0.05 + index * 0.01)


def fixture_for(conv, sampled_out_nodes=None, with_softmax=True):
    torch.manual_seed(0)
    if sampled_out_nodes:
        conv.sampled_out_nodes = sampled_out_nodes
    data = Data()
    model = Model(conv, with_softmax=with_softmax)
    initialize_model(model)

    visualizer = GNNVisualizer()
    visualizer.add_model(
        data=data,
        model=model,
        subgraphSample=False,
        queries=[[1, 3]],
        mode="node",
    )
    return {
        "graphData": visualizer.graphData,
        "intmData": visualizer.intmData,
        "modelInfo": visualizer.modelInfo,
        "queries": visualizer.queries,
        "subgraphSample": visualizer.subgraphSample,
        "mode": visualizer.mode,
    }


def main():
    output_dir = Path(__file__).parent / ".cache" / "fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "gcn_logits": fixture_for(GCNConv(3, 2), with_softmax=False),
        "gat": fixture_for(GATConv(3, 2, heads=2, concat=True)),
        "graphsage": fixture_for(SAGEConv(3, 2), sampled_out_nodes=[2]),
        "gin": fixture_for(GINConv(torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh()))),
    }

    for name, fixture in fixtures.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
