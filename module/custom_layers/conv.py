import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers, utils
from torchdrug.layers import functional


class CompositionalGraphConv(layers.MessagePassingBase):

    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None, batch_norm=False, activation="relu",
                 composition="multiply"):
        super(CompositionalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.edge_input_dim = edge_input_dim
        self.composition = composition

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relation = nn.Embedding(num_relation, input_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        node_in, _, relation = graph.edge_list.t()
        node_in_feature = input[node_in]
        relation_feature = self.relation.weight[relation]
        if self.composition == "subtract":
            message = node_in_feature - relation_feature
        elif self.composition == "multiply":
            message = node_in_feature * relation_feature
        else:
            raise ValueError("Composition method `%s` is not supported." % self.composition)
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node) / \
                 (scatter_add(edge_weight, node_out, dim=0, dim_size=graph.num_node) + self.eps)
        return update

    def combine(self, input, update):
        output = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
