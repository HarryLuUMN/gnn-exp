import json
from pathlib import Path

import torch

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


class GATConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 4)
        self.bias = torch.nn.Parameter(torch.zeros(4))
        self.heads = 2
        self.concat = True
        self.att_src = torch.nn.Parameter(torch.zeros(1, 2, 2))
        self.att_dst = torch.nn.Parameter(torch.zeros(1, 2, 2))

    def forward(self, x, edge_index):
        return self.lin(x) + self.bias


class SAGEConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_l = torch.nn.Linear(3, 2)
        self.lin_r = torch.nn.Linear(3, 2, bias=False)
        self.aggr = "mean"

    def forward(self, x, edge_index):
        return self.lin_l(x)


class GINConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh())
        self.eps = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index):
        return self.nn(x)


class Model(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv1 = conv
        hidden_dim = 4 if type(conv).__name__ == "GATConv" else 2
        self.act1 = torch.nn.Tanh()
        self.classifier = torch.nn.Linear(hidden_dim, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        h = self.act1(self.conv1(x, edge_index))
        return self.softmax(self.classifier(h))


def initialize_linear(linear, offset):
    with torch.no_grad():
        values = torch.arange(linear.weight.numel(), dtype=torch.float32)
        linear.weight.copy_(values.reshape_as(linear.weight) * 0.03 + offset)
        if linear.bias is not None:
            bias = torch.arange(linear.bias.numel(), dtype=torch.float32)
            linear.bias.copy_(bias * 0.02 - offset)


def initialize_model(model):
    for index, module in enumerate(model.modules()):
        if isinstance(module, torch.nn.Linear):
            initialize_linear(module, offset=0.05 + index * 0.01)


def fixture_for(conv):
    torch.manual_seed(0)
    data = Data()
    model = Model(conv)
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
        "gat": fixture_for(GATConv()),
        "graphsage": fixture_for(SAGEConv()),
        "gin": fixture_for(GINConv()),
    }

    for name, fixture in fixtures.items():
        path = output_dir / f"{name}.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
