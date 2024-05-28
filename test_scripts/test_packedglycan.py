import os
import sys
import argparse
from matplotlib import pyplot as plt

from glycowork.glycan_data.loader import *
from glycowork.glycan_data.data_entry import *
from glycowork.motif.draw import GlycoDraw

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module.custom_data import glycan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="output direction", required=True)
    parser.add_argument("--batch_size", type=int, default=4, help="number of glycans in each batch")
    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    half_batch_size = batch_size // 2
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_samples = df_species["glycan"]
    connected_samples = [sample for sample in all_samples if "{" not in sample and "}" not in sample]
    unconnected_samples = [sample for sample in all_samples if "{" in sample or "}" in sample]

    # Method 1: Construct a batch from a list of glycan sequences
    iupacs = connected_samples[:half_batch_size] + unconnected_samples[:half_batch_size]
    graphs = glycan.PackedGlycan.from_iupac(iupacs)
    print(graphs)
    print("Node number: ", graphs.num_nodes, graphs.num_node)
    print("Node feature: ", graphs.node_feature.shape)
    print("Edge number: ", graphs.num_edges, graphs.num_edge)
    print("Edge feature: ", graphs.edge_feature.shape)
    print("Glycoword number: ", graphs.num_glycowords, graphs.num_glycoword)
    print("Glycoword type: ", graphs.glycoword_type.shape)
    print("")
    torchglycan_file = os.path.join(output_dir, "packed_glycan.pdf")
    graphs.visualize(save_file=torchglycan_file)

    # Method 2: Construct a batch from a list of glycan instances
    graphs = []
    for iupac in iupacs:
        graph = glycan.Glycan.from_iupac(iupac)
        graphs.append(graph)
    graphs = glycan.Glycan.pack(graphs)
    print(graphs)
    print("Node number: ", graphs.num_nodes, graphs.num_node)
    print("Node feature: ", graphs.node_feature.shape)
    print("Edge number: ", graphs.num_edges, graphs.num_edge)
    print("Edge feature: ", graphs.edge_feature.shape)
    print("Glycoword number: ", graphs.num_glycowords, graphs.num_glycoword)
    print("Glycoword type: ", graphs.glycoword_type.shape)
