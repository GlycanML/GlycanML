import re
import math
import copy
import warnings
import numpy as np
import pandas as pd
from collections import Sequence, defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean

from torchdrug import core, layers, utils
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from torchdrug.utils import pretty

from module.custom_data.glycan import Glycan, PackedGlycan
import module.custom_models.readout as glycan_readout


#        Change readout.py with self.type == 'glycan' to enable readout        #


@R.register("models.GlycanConvolutionalNetwork")
class GlycanConvolutionalNetwork(nn.Module, core.Configurable):
    """
    Transfered from Protein Shallow CNN proposed in `Is Transfer Learning Necessary for Protein Landscape Prediction?`_.

    .. _Is Transfer Learning Necessary for Protein Landscape Prediction?:
        https://arxiv.org/pdf/2011.03443.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        glycoword_dim (int): number of glycowords
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
        short_cut (bool, optional): use short cut or not
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum``, ``mean``, ``max`` and ``attention``.
    """

    def __init__(self, input_dim, hidden_dims, glycoword_dim, kernel_size=3, stride=1, padding=1,
                activation='relu', short_cut=False, concat_hidden=False, readout="max"):
        super(GlycanConvolutionalNetwork, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                nn.Conv1d(self.dims[i], self.dims[i+1], kernel_size, stride, padding)
            )

        # change
        if readout == "sum":
            self.readout = glycan_readout.SumReadout('glycan')
        elif readout == "mean":
            self.readout = glycan_readout.MeanReadout('glycan')
        elif readout == "max":
            self.readout = glycan_readout.MaxReadout('glycan')
        elif readout == "attention":
            self.readout = glycan_readout.AttentionReadout(self.output_dim, 'glycan')
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Forward.
        """
        # not precise
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)
        input = functional.variadic_to_padded(input, graph.num_glycowords, value=self.padding_id)[0]

        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            # change
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2)
            hidden = self.activation(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        # change
        glycoword_feature = functional.padded_to_variadic(hidden, graph.num_glycowords)
        graph_feature = self.readout(graph, glycoword_feature)
        
        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


@R.register("models.GlycanResNet")
class GlycanResNet(nn.Module, core.Configurable):
    """
    Transfered from Protein ResNet proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        glycoword_dim (int): number of glycowords
        kernel_size (int, optional): size of convolutional kernel
        stride (int, optional): stride of convolution
        padding (int, optional): padding added to both sides of the input
        activation (str or function, optional): activation function
        short_cut (bool, optional): use short cut or not
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        layer_norm (bool, optional): apply layer normalization or not
        dropout (float, optional): dropout ratio of input features
        readout (str, optional): readout function. Available functions are ``sum``, ``mean`` and ``attention``.
    """

    def __init__(self, input_dim, hidden_dims, glycoword_dim, kernel_size=3, stride=1, padding=1,
                 activation="gelu", short_cut=False, concat_hidden=False, layer_norm=False,
                 dropout=0, readout="attention"):
        super(GlycanResNet, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.padding_id = input_dim - 1

        self.embedding = nn.Linear(input_dim, hidden_dims[0])
        self.position_embedding = layers.SinusoidalPositionEmbedding(hidden_dims[0])
        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dims[0])
        else:
            self.layer_norm = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.ProteinResNetBlock(self.dims[i], self.dims[i + 1], kernel_size,
                                                         stride, padding, activation))

        if readout == "sum":
            self.readout = glycan_readout.SumReadout("glycan")
        elif readout == "mean":
            self.readout = glycan_readout.MeanReadout("glycan")
        elif readout == "attention":
            self.readout = glycan_readout.AttentionReadout(self.output_dim, "glycan")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Forward.
        """
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)
        input, mask = functional.variadic_to_padded(input, graph.num_glycowords, value=self.padding_id)
        mask = mask.unsqueeze(-1)

        input = self.embedding(input) + self.position_embedding(input).unsqueeze(0)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)
        input = input * mask
        
        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            hidden = layer(layer_input, mask)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            hidden = torch.cat(hiddens, dim=-1)
        else:
            hidden = hiddens[-1]

        glycoword_feature = functional.padded_to_variadic(hidden, graph.num_glycowords)
        graph_feature = self.readout(graph, glycoword_feature)
        
        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


@R.register("models.GlycanLSTM")
class GlycanLSTM(nn.Module, core.Configurable):
    """
    Transfered from Protein LSTM proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int, optional): hidden dimension
        glycoword_dim (int): number of glycowords
        num_layers (int, optional): number of LSTM layers
        activation (str or function, optional): activation function
        layer_norm (bool, optional): apply layer normalization or not
        dropout (float, optional): dropout ratio of input features
    """

    def __init__(self, input_dim, hidden_dim, glycoword_dim, num_layers, activation='tanh', layer_norm=False, 
                dropout=0):
        super(GlycanLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim    # output_dim for node feature is 2 * hidden_dim
        self.node_output_dim = 2 * hidden_dim
        self.num_layers = num_layers
        self.padding_id = input_dim - 1

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_init = nn.Embedding(glycoword_dim, input_dim)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            bidirectional=True)

        self.reweight = nn.Linear(2 * num_layers, 1)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Forward.
        """
        input = graph.glycoword_type.long()
        input = self.embedding_init(input)
        input = functional.variadic_to_padded(input, graph.num_glycowords, value=self.padding_id)[0]
        
        input = self.embedding(input)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)

        output, hidden = self.lstm(input)

        glycoword_feature = functional.padded_to_variadic(output, graph.num_glycowords)

        # (2 * num_layer, B, d)
        graph_feature = self.reweight(hidden[0].permute(1, 2, 0)).squeeze(-1)
        graph_feature = self.linear(graph_feature)
        graph_feature = self.activation(graph_feature)

        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }


@R.register("models.GlycanBERT")
class GlycanBERT(nn.Module, core.Configurable):
    """
    Transfered from Protein BERT proposed in `Evaluating Protein Transfer Learning with TAPE`_.

    .. _Evaluating Protein Transfer Learning with TAPE:
        https://arxiv.org/pdf/1906.08230.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int, optional): hidden dimension
        num_layers (int, optional): number of Transformer blocks
        num_heads (int, optional): number of attention heads
        intermediate_dim (int, optional): intermediate hidden dimension of Transformer block
        activation (str or function, optional): activation function
        hidden_dropout (float, optional): dropout ratio of hidden features
        attention_dropout (float, optional): dropout ratio of attention maps
        max_position (int, optional): maximum number of positions
    """

    def __init__(self, input_dim, hidden_dim=768, num_layers=12, num_heads=12, intermediate_dim=3072,
                 activation="gelu", hidden_dropout=0.1, attention_dropout=0.1, max_position=8192):
        super(GlycanBERT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position = max_position

        self.num_glycoword_type = input_dim
        self.embedding = nn.Embedding(input_dim + 3, hidden_dim)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(layers.ProteinBERTBlock(hidden_dim, intermediate_dim, num_heads,
                                                       attention_dropout, hidden_dropout, activation))
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Forward.
        """
        input = graph.glycoword_type
        size_ext = graph.num_glycowords
        # Prepend BOS
        bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.num_glycoword_type
        input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        # Append EOS
        eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * (self.num_glycoword_type + 1)
        input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        # Padding
        input, mask = functional.variadic_to_padded(input, size_ext, value=self.num_glycoword_type + 2)
        mask = mask.long().unsqueeze(-1)

        input = self.embedding(input)
        position_indices = torch.arange(input.shape[1], device=input.device)
        input = input + self.position_embedding(position_indices).unsqueeze(0)
        input = self.layer_norm(input)
        input = self.dropout(input)

        for layer in self.layers:
            input = layer(input, mask)

        glycoword_feature = functional.padded_to_variadic(input, graph.num_glycowords)

        graph_feature = input[:, 0]
        graph_feature = self.linear(graph_feature)
        graph_feature = F.tanh(graph_feature)

        return {
            "graph_feature": graph_feature,
            "glycoword_feature": glycoword_feature
        }
