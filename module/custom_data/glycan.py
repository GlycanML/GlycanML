import re
import os
import math
import copy
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Sequence, defaultdict

import torch
from torch_scatter import scatter_add, scatter_mean
from glycowork.motif import tokenization

from torchdrug import utils
from torchdrug.data import constant, Graph, PackedGraph, Molecule, PackedMolecule
from torchdrug.core import Registry as R
from torchdrug.utils import pretty

from module.custom_data import glycan_feature


class Glycan(Graph):
    """
    Glycans with sequence and graph views.
    """

    _meta_types = {"node", "edge", "glycoword", "graph",
                   "node reference", "edge reference", "graph reference"}

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "glycoword_vocab.pkl")
    entities = pkl.load(open(data_path, "rb"))
    units = [entity for entity in entities
             if not (entity.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", entity))]
    unit2id = {x: i for i, x in enumerate(units)}
    id2unit = {i: x for x, i in unit2id.items()}
    links = [entity for entity in entities
             if entity.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", entity)]
    link2id = {x: i for i, x in enumerate(links)}
    id2link = {i: x for x, i in link2id.items()}
    glycowords = entities + ["[", "]", "{", "}", "Unknown_Token"]
    glycoword2id = {x: i for i, x in enumerate(glycowords)}
    id2glycoword = {i: x for x, i in glycoword2id.items()}

    def __init__(self, edge_list=None, unit_type=None, link_type=None, glycoword_type=None,
                 unit_feature=None, link_feature=None, glycan_feature=None, **kwargs):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.links)
        super(Glycan, self).__init__(edge_list=edge_list, **kwargs)
        unit_type, link_type = self._standarize_unit_link(unit_type, link_type)
        glycoword_type, num_glycoword = self._standarize_num_glycoword(glycoword_type)
        self.num_glycoword = num_glycoword

        with self.unit():
            if unit_feature is not None:
                self.unit_feature = torch.as_tensor(unit_feature, device=self.device)
            self.unit_type = unit_type

        with self.link():
            if link_feature is not None:
                self.link_feature = torch.as_tensor(link_feature, device=self.device)
            self.link_type = link_type

        with self.glycan():
            if glycan_feature is not None:
                self.glycan_feature = torch.as_tensor(glycan_feature, device=self.device)

        if self.num_glycoword >= 0:
            with self.glycoword():
                self.glycoword_type = glycoword_type

    def _standarize_unit_link(self, unit_type, link_type):
        if unit_type is None:
            raise ValueError("`unit_type` should be provided")
        if link_type is None:
            raise ValueError("`link_type` should be provided")

        unit_type = torch.as_tensor(unit_type, dtype=torch.long, device=self.device)
        link_type = torch.as_tensor(link_type, dtype=torch.long, device=self.device)
        return unit_type, link_type

    def _standarize_num_glycoword(self, glycoword_type):
        if glycoword_type is not None:
            glycoword_type = torch.as_tensor(glycoword_type, dtype=torch.long, device=self.device)
            num_glycoword = torch.tensor(len(glycoword_type), device=self.device)
        else:
            num_glycoword = torch.tensor(-1, device=self.device)

        return glycoword_type, num_glycoword

    @classmethod
    def _standarize_option(cls, option):
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    @classmethod
    def multireplace(cls, string, replace_dict):
        for k, v in replace_dict.items():
            string = string.replace(k, v)
        return string

    @classmethod
    def get_unit_id2start_end(cls, iupac):
        units_links = [x for x in cls.multireplace(iupac, {"[": "", "]": "", ")": "("}).split("(") if x]
        units = [x for x in units_links
                 if not (x.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", x))]
        id2unit = {i: x for i, x in enumerate(units)}
        id2start_end = {}
        curr_position = 0
        for unit_id, unit in id2unit.items():
            start = iupac[curr_position:].index(unit) + curr_position
            id2start_end[unit_id] = [start, start + len(unit)]
            curr_position = start + len(unit)

        return id2start_end

    @classmethod
    def locate_brackets(cls, iupac):
        left_indices = []
        right_indices = []
        for i, x in enumerate(iupac):
            if x == "[":
                left_indices.append(i)
            if x == "]" and len(left_indices) > len(right_indices):
                right_indices.append(i)

        return left_indices, right_indices

    @classmethod
    def on_branch(cls, index, left_indices, right_indices):
        in_intervals = [index >= left_index and index <= right_index
                        for left_index, right_index in zip(left_indices, right_indices)]
        return any(in_intervals)

    @classmethod
    def get_link_type(cls, iupac):
        left_indices, right_indices = cls.locate_brackets(iupac)
        if len(left_indices) == 0 and len(right_indices) == 0:
            link = "".join([x for x in iupac if x not in ["(", ")", "[", "]"]])
            link_type = cls.link2id.get(tokenization.get_core(link), None)
        elif len(left_indices) != len(right_indices):
            link_type = None
        else:
            link = "".join([x for i, x in enumerate(iupac)
                            if x not in ["(", ")", "[", "]"] and not cls.on_branch(i, left_indices, right_indices)])
            link_type = cls.link2id.get(tokenization.get_core(link), None)

        return link_type

    @classmethod
    def get_edge_list(cls, iupac):
        parts = [x for x in cls.multireplace(iupac, {"}": "{"}).split("{") if x]
        edge_list = []
        num_cum_unit = 0
        for part in parts:
            unit_id2start_end = cls.get_unit_id2start_end(part)
            num_unit = len(unit_id2start_end)
            for src_id in range(num_unit):
                for tgt_id in range(src_id + 1, num_unit):
                    src_end = unit_id2start_end[src_id][1]
                    tgt_start = unit_id2start_end[tgt_id][0]
                    assert tgt_start > src_end
                    inner_part = part[src_end:tgt_start]
                    link_type = cls.get_link_type(inner_part)
                    if link_type is not None:
                        src = src_id + num_cum_unit
                        tgt = tgt_id + num_cum_unit
                        edge_list += [[src, tgt, link_type], [tgt, src, link_type]]
            num_cum_unit += num_unit

        return edge_list

    @classmethod
    def locate_glycowords(cls, iupac, glycowords):
        glycoword_location = []
        curr_position = 0
        for glycoword in glycowords:
            start = iupac[curr_position:].index(glycoword) + curr_position
            end = start + len(glycoword)
            glycoword_location.append([glycoword, start, end])
            curr_position = end

        return glycoword_location

    @classmethod
    def from_iupac(cls, iupac, unit_feature="default", link_feature="default", glycan_feature=None):
        """
        Create a glycan from its IUPAC-condensed sequence.
        """
        # construct the graph view
        unit_feature = cls._standarize_option(unit_feature)
        link_feature = cls._standarize_option(link_feature)
        glycan_feature = cls._standarize_option(glycan_feature)

        unit_type = []
        _unit_feature = []
        units_links = [x for x in cls.multireplace(iupac,
                                                   {"[": "", "]": "", "{": "", "}": "", ")": "("}).split("(") if x]
        units = [x for x in units_links
                 if not (x.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", x))]
        for unit in units:
            unit_type.append(cls.unit2id[tokenization.get_core(unit)])
            feature = []
            for name in unit_feature:
                func = R.get("features.unit.%s" % name)
                feature += func(unit)
            _unit_feature.append(feature)
        unit_type = torch.tensor(unit_type)
        if len(unit_feature) > 0:
            _unit_feature = torch.tensor(_unit_feature)
        else:
            _unit_feature = None

        edge_list = cls.get_edge_list(iupac)
        link_type = []
        _link_feature = []
        for edge in edge_list:
            src_, tgt_, type_ = edge
            link_type.append(type_)
            feature = []
            for name in link_feature:
                func = R.get("features.link.%s" % name)
                feature += func(cls.id2link[type_])
            _link_feature.append(feature)
        link_type = torch.tensor(link_type)
        if len(link_feature) > 0:
            _link_feature = torch.tensor(_link_feature)
        else:
            _link_feature = None

        _glycan_feature = []
        for name in glycan_feature:
            func = R.get("features.glycan.%s" % name)
            _glycan_feature += func(iupac)
        if len(glycan_feature) > 0:
            _glycan_feature = torch.tensor(_glycan_feature)
        else:
            _glycan_feature = None

        # construct the sequence view
        glycoword_location = cls.locate_glycowords(iupac, units_links)
        glycoword_type = []
        last_start, last_end = 0, 0
        for glycoword, start, end in glycoword_location:
            for letter in iupac[last_end:start]:
                if letter in ["[", "]", "{", "}"]:
                    glycoword_type.append(cls.glycoword2id[letter])
            glycoword_type.append(cls.glycoword2id.get(tokenization.get_core(glycoword), len(cls.glycowords) - 1))
            last_start, last_end = start, end
        for letter in iupac[last_end:]:
            if letter in ["[", "]", "{", "}"]:
                glycoword_type.append(cls.glycoword2id[letter])
        glycoword_type = torch.tensor(glycoword_type)

        num_relation = len(cls.links)
        return cls(edge_list, unit_type, link_type, glycoword_type,
                   unit_feature=_unit_feature, link_feature=_link_feature, glycan_feature=_glycan_feature,
                   num_node=len(units), num_relation=num_relation)

    @classmethod
    def pack(cls, graphs):
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        num_glycowords = []
        num_relation = -1
        num_cum_node = 0
        num_cum_edge = 0
        num_cum_glycoword = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            num_glycowords.append(graph.num_glycoword)
            for k, v in graph.data_dict.items():
                for type in meta_dict[k]:
                    if type == "graph":
                        v = v.unsqueeze(0)
                    elif type == "node reference":
                        v = v + num_cum_node
                    elif type == "edge reference":
                        v = v + num_cum_edge
                    elif type == "graph reference":
                        v = v + num_graph
                data_dict[k].append(v)
            if num_relation == -1:
                num_relation = graph.num_relation
            elif num_relation != graph.num_relation:
                raise ValueError("Inconsistent `num_relation` in graphs. Expect %d but got %d."
                                 % (num_relation, graph.num_relation))
            num_cum_node += graph.num_node
            num_cum_edge += graph.num_edge
            num_cum_glycoword += graph.num_glycoword
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}

        return cls.packed_type(edge_list, edge_weight=edge_weight, num_relation=num_relation,
                               num_nodes=num_nodes, num_edges=num_edges, num_glycowords=num_glycowords,
                               meta_dict=meta_dict, **data_dict)

    def undirected(self, add_inverse=False):
        if add_inverse:
            raise ValueError("Links are undirected relations, but `add_inverse` is specified")
        return super(Glycan, self).undirected(add_inverse)

    def unit(self):
        """
        Context manager for unit attributes.
        """
        return self.node()

    def link(self):
        """
        Context manager for link attributes.
        """
        return self.edge()

    def glycoword(self):
        """
        Context manager for glycoword attributes.
        """
        return self.context("glycoword")

    def glycan(self):
        """
        Context manager for glycan attributes.
        """
        return self.graph()

    def unit_reference(self):
        """
        Context manager for unit references.
        """
        return self.node_reference()

    def link_reference(self):
        """
        Context manager for link references.
        """
        return self.edge_reference()

    def glycan_reference(self):
        """
        Context mangaer for glycan references.
        """
        return self.graph_reference()

    @property
    def num_node(self):
        return self.num_unit

    @num_node.setter
    def num_node(self, value):
        self.num_unit = value

    @property
    def num_edge(self):
        return self.num_link

    @num_edge.setter
    def num_edge(self, value):
        self.num_link = value

    unit2graph = Graph.node2graph
    link2graph = Graph.edge2graph

    @property
    def node_feature(self):
        return self.unit_feature

    @node_feature.setter
    def node_feature(self, value):
        self.unit_feature = value

    @property
    def edge_feature(self):
        return self.link_feature

    @edge_feature.setter
    def edge_feature(self, value):
        self.link_feature = value

    @property
    def graph_feature(self):
        return self.glycan_feature

    @graph_feature.setter
    def graph_feature(self, value):
        self.glycan_feature = value

    def _check_attribute(self, key, value):
        super(Glycan, self)._check_attribute(key, value)
        for type in self._meta_contexts:
            if type == "glycoword":
                if len(value) != self.num_glycoword:
                    raise ValueError("Expect glycoword attribute `%s` to have shape (%d, *), but found %s" %
                                     (key, self.num_glycoword, value.shape))

    @property
    def glycoword2graph(self):
        """Glycoword id to graph id mapping."""
        if self.num_glycoword < 0:
            raise ValueError("Glycowords are not specified for this glycan.")
        return torch.zeros(self.num_glycoword, dtype=torch.long, device=self.device)

    def __repr__(self):
        fields = ["num_unit=%d" % self.num_unit, "num_link=%d" % self.num_link]
        if self.num_glycoword >= 0:
            fields.append("num_glycoword=%d" % self.num_glycoword)
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


class PackedGlycan(PackedGraph, Glycan):
    """
    Container for glycans with variadic sizes.
    """

    unpacked_type = Glycan
    unit2graph = PackedGraph.node2graph
    link2graph = PackedGraph.edge2graph
    _check_attribute = Glycan._check_attribute

    def __init__(self, edge_list=None, unit_type=None, link_type=None, glycoword_type=None, num_nodes=None,
                 num_edges=None, num_glycowords=None, offsets=None, **kwargs):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.links)
        super(PackedGlycan, self).__init__(edge_list=edge_list, num_nodes=num_nodes, num_edges=num_edges,
                                           offsets=offsets, unit_type=unit_type, link_type=link_type,
                                           glycoword_type=glycoword_type, **kwargs)

        if num_glycowords is not None:
            num_glycowords = torch.as_tensor(num_glycowords, device=self.device)
            num_cum_glycowords = num_glycowords.cumsum(0)
            self.num_glycowords = num_glycowords
            self.num_cum_glycowords = num_cum_glycowords
        else:
            self.num_glycowords, self.num_cum_glycowords = None, None

    @classmethod
    def from_iupac(cls, iupacs, unit_feature="default", link_feature="default", glycan_feature=None):
        """
        Create a packed glycan from a list of IUPAC-condensed glycan sequences.
        """
        unit_feature = cls._standarize_option(unit_feature)
        link_feature = cls._standarize_option(link_feature)
        glycan_feature = cls._standarize_option(glycan_feature)

        unit_type = []
        _unit_feature = []
        edge_list = []
        link_type = []
        _link_feature = []
        glycoword_type = []
        _glycan_feature = []
        num_nodes = []
        num_edges = []
        num_glycowords = []

        for iupac in iupacs:
            # construct the graph view
            units_links = [x for x in
                           cls.multireplace(iupac, {"[": "", "]": "", "{": "", "}": "", ")": "("}).split("(") if x]
            units = [x for x in units_links
                     if not (x.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", x))]
            for unit in units:
                unit_type.append(cls.unit2id[tokenization.get_core(unit)])
                feature = []
                for name in unit_feature:
                    func = R.get("features.unit.%s" % name)
                    feature += func(unit)
                _unit_feature.append(feature)

            _edge_list = cls.get_edge_list(iupac)
            edge_list += _edge_list
            for edge in _edge_list:
                src_, tgt_, type_ = edge
                link_type.append(type_)
                feature = []
                for name in link_feature:
                    func = R.get("features.link.%s" % name)
                    feature += func(cls.id2link[type_])
                _link_feature.append(feature)

            feature = []
            for name in glycan_feature:
                func = R.get("features.glycan.%s" % name)
                feature += func(iupac)
            _glycan_feature.append(feature)

            # construct the sequence view
            glycoword_location = cls.locate_glycowords(iupac, units_links)
            glycoword_cnt = 0
            last_start, last_end = 0, 0
            for glycoword, start, end in glycoword_location:
                for letter in iupac[last_end:start]:
                    if letter in ["[", "]", "{", "}"]:
                        glycoword_type.append(cls.glycoword2id[letter])
                        glycoword_cnt += 1
                glycoword_type.append(cls.glycoword2id.get(tokenization.get_core(glycoword), len(cls.glycowords) - 1))
                glycoword_cnt += 1
                last_start, last_end = start, end
            for letter in iupac[last_end:]:
                if letter in ["[", "]", "{", "}"]:
                    glycoword_type.append(cls.glycoword2id[letter])
                    glycoword_cnt += 1

            num_nodes.append(len(units))
            num_edges.append(len(_edge_list))
            num_glycowords.append(glycoword_cnt)

        unit_type = torch.tensor(unit_type)
        edge_list = torch.tensor(edge_list)
        link_type = torch.tensor(link_type)
        glycoword_type = torch.tensor(glycoword_type)
        _unit_feature = torch.tensor(_unit_feature) if len(unit_feature) > 0 else None
        _link_feature = torch.tensor(_link_feature) if len(link_feature) > 0 else None
        _glycan_feature = torch.tensor(_glycan_feature) if len(glycan_feature) > 0 else None

        num_relation = len(cls.links)
        return cls(edge_list, unit_type, link_type, glycoword_type,
                   unit_feature=_unit_feature, link_feature=_link_feature, glycan_feature=_glycan_feature,
                   num_nodes=num_nodes, num_edges=num_edges, num_glycowords=num_glycowords,
                   num_relation=num_relation)

    def undirected(self, add_inverse=False):
        if add_inverse:
            raise ValueError("Links are undirected relations, but `add_inverse` is specified")
        return super(PackedGlycan, self).undirected(add_inverse)

    @property
    def num_nodes(self):
        return self.num_units

    @num_nodes.setter
    def num_nodes(self, value):
        self.num_units = value

    @property
    def num_edges(self):
        return self.num_links

    @num_edges.setter
    def num_edges(self, value):
        self.num_links = value

    @utils.cached_property
    def glycoword2graph(self):
        """Glycoword id to graph id mapping."""
        if self.num_glycowords is None:
            raise ValueError("Glycowords are not specified for this batch of glycans.")
        range = torch.arange(self.batch_size, device=self.device)
        glycoword2graph = range.repeat_interleave(self.num_glycowords)
        return glycoword2graph

    def cuda(self, *args, **kwargs):
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges, num_glycowords=self.num_glycowords,
                              num_relation=self.num_relation, offsets=self._offsets, meta_dict=self.meta_dict,
                              **utils.cuda(self.data_dict, *args, **kwargs))

    def cpu(self):
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges, num_glycowords=self.num_glycowords,
                              num_relation=self.num_relation, offsets=self._offsets, meta_dict=self.meta_dict,
                              **utils.cpu(self.data_dict))

    def __repr__(self):
        fields = ["batch_size=%d" % self.batch_size,
                  "num_units=%s" % pretty.long_array(self.num_units.tolist()),
                  "num_links=%s" % pretty.long_array(self.num_links.tolist())]
        if self.num_glycowords is not None:
            fields.append("num_glycowords=%s" % pretty.long_array(self.num_glycowords.tolist()))
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


Glycan.packed_type = PackedGlycan
