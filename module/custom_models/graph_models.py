from collections.abc import Sequence

import torch
from torch import nn

from torchdrug import core, layers, models
from torchdrug.core import Registry as R

from module.custom_layers import CompGCNConv


@R.register("models.GlycanGCN")
class GlycanGCN(models.GCN):
    """
    Graph Convolutional Network proposed in `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):    
        super(GlycanGCN, self).__init__(input_dim, hidden_dims, edge_input_dim, short_cut, batch_norm, 
                                        activation, concat_hidden, readout.replace("dual", "mean"))
        
        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.output_dim = self.output_dim * 2

    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGCN, self).forward(graph, input, all_loss, metric)

        if hasattr(self, "readout_ext"):
            node_feature, graph_feature = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)
                
        return feature


@R.register("models.GlycanRGCN")
class GlycanRGCN(models.RGCN):
    """
    Relational Graph Convolutional Network proposed in `Modeling Relational Data with Graph Convolutional Networks?`_.

    .. _Modeling Relational Data with Graph Convolutional Networks?:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, num_relation, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanRGCN, self).__init__(input_dim, hidden_dims, num_relation, edge_input_dim, short_cut, batch_norm, 
                                        activation, concat_hidden, readout.replace("dual", "mean"))
        
        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.output_dim = self.output_dim * 2

    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanRGCN, self).forward(graph, input, all_loss, metric)

        if hasattr(self, "readout_ext"):
            node_feature, graph_feature = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)
                
        return feature
    

@R.register("models.GlycanGAT")
class GlycanGAT(models.GAT):
    """
    Graph Attention Network proposed in `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None, num_head=1, negative_slope=0.2, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanGAT, self).__init__(input_dim, hidden_dims, edge_input_dim, num_head, negative_slope, short_cut, batch_norm, 
                                        activation, concat_hidden, readout.replace("dual", "mean"))
        
        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.output_dim = self.output_dim * 2

    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGAT, self).forward(graph, input, all_loss, metric)

        if hasattr(self, "readout_ext"):
            node_feature, graph_feature = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)
                
        return feature
    

@R.register("models.GlycanGIN")
class GlycanGIN(models.GIN):
    """
    Graph Ismorphism Network proposed in `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        num_mlp_layer (int, optional): number of MLP layers
        eps (int, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
    """

    def __init__(self, input_dim, hidden_dims, num_unit, edge_input_dim=None, num_mlp_layer=2, eps=0, learn_eps=False,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GlycanGIN, self).__init__(input_dim, hidden_dims, edge_input_dim, num_mlp_layer, eps, learn_eps, 
                                        short_cut, batch_norm, activation, concat_hidden, readout.replace("dual", "mean"))
        
        self.embedding = nn.Embedding(num_unit, input_dim)
        if readout == "dual":
            self.readout_ext = layers.MaxReadout()
            self.output_dim = self.output_dim * 2

    def forward(self, graph, input, all_loss=None, metric=None):
        input = self.embedding(graph.unit_type)
        feature = super(GlycanGIN, self).forward(graph, input, all_loss, metric)

        if hasattr(self, "readout_ext"):
            node_feature, graph_feature = feature["node_feature"], feature["graph_feature"]
            feature["graph_feature"] = torch.cat([graph_feature, self.readout_ext(graph, node_feature)], dim=-1)
                
        return feature


@R.register("models.GlycanCompGCN")
class GlycanCompGCN(nn.Module, core.Configurable):
    """
    CompGCN proposed in `Composition-based Multi-Relational Graph Convolutional Networks`_

    .. _Composition-based Multi-Relational Graph Convolutional Networks:
        https://arxiv.org/pdf/1911.03082.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        num_unit (int): number of monosaccharide units
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``dual``.
        composition (str, optional): composition method. Available functions are ``multiply`` and ``subtract``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, num_unit, edge_input_dim=None, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum", composition="multiply"):
        super(GlycanCompGCN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.embedding_init = nn.Embedding(num_unit, input_dim)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(CompGCNConv(self.dims[i], self.dims[i + 1], num_relation, edge_input_dim,
                                           batch_norm, activation, composition))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "max":
            self.readout = layers.MaxReadout()
        elif readout == "dual":
            self.readout1, self.readout2 = layers.MeanReadout(), layers.MaxReadout()
            self.output_dim = self.output_dim * 2
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        hiddens = []
        layer_input = self.embedding_init(graph.unit_type)

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        if hasattr(self, "readout1"):
            graph_feature = torch.cat([self.readout1(graph, node_feature), self.readout2(graph, node_feature)], dim=-1)
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }

@R.register("models.GlycanMPNN")
class GlycanMPNN(nn.Module, core.Configurable):
    """
    Message Passing Neural Network proposed in `Neural Message Passing for Quantum Chemistry`_.

    This implements the enn-s2s variant in the original paper.

    .. _Neural Message Passing for Quantum Chemistry:
        https://arxiv.org/pdf/1704.01212.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        num_unit (int): number of monosaccharide units
        edge_input_dim (int): dimension of edge features
        num_layer (int, optional): number of hidden layers
        num_gru_layer (int, optional): number of GRU layers in each node update
        num_mlp_layer (int, optional): number of MLP layers in each message function
        num_s2s_step (int, optional): number of processing steps in set2set
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
    """

    def __init__(self, input_dim, hidden_dim, num_unit, edge_input_dim, num_layer=1, num_gru_layer=1, num_mlp_layer=2,
                 num_s2s_step=3, short_cut=False, batch_norm=False, activation="relu", concat_hidden=False):
        super(GlycanMPNN, self).__init__()

        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        if concat_hidden:
            feature_dim = hidden_dim * num_layer
        else:
            feature_dim = hidden_dim
        self.output_dim = feature_dim * 2
        self.node_output_dim = feature_dim
        self.num_layer = num_layer
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.embedding_init = nn.Embedding(num_unit, hidden_dim)
        self.layer = layers.MessagePassing(hidden_dim, edge_input_dim, [hidden_dim] * (num_mlp_layer - 1),
                                           batch_norm, activation)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_gru_layer)

        self.readout = layers.Set2Set(feature_dim, num_step=num_s2s_step)

    def forward(self, graph, input, all_loss=None, metric=None):
        hiddens = []
        layer_input = self.embedding_init(graph.unit_type)
        hx = layer_input.repeat(self.gru.num_layers, 1, 1)

        for i in range(self.num_layer):
            x = self.layer(graph, layer_input)
            hidden, hx = self.gru(x.unsqueeze(0), hx)
            hidden = hidden.squeeze(0)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }