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
    parser.add_argument("--num_sample", type=int, default=3, help="number of test samples for each category")
    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_samples = df_species["glycan"]
    connected_samples = [sample for sample in all_samples if "{" not in sample and "}" not in sample]
    unconnected_samples = [sample for sample in all_samples if "{" in sample or "}" in sample]

    for i in range(args.num_sample):
        sample = connected_samples[i]
        print(sample)
        glycowork_file = os.path.join(output_dir, "connected_glycowork_%d.pdf" % i)
        GlycoDraw(sample, filepath=glycowork_file)
        graph = glycan.Glycan.from_iupac(sample)
        print(graph)
        print("#glycowords: %d\n" % len(graph.glycoword_type))
        torchglycan_file = os.path.join(output_dir, "connected_torchglycan_%d.pdf" % i)
        graph.visualize(save_file=torchglycan_file)

        sample = unconnected_samples[i]
        print(sample)
        glycowork_file = os.path.join(output_dir, "unconnected_glycowork_%d.pdf" % i)
        GlycoDraw(sample, filepath=glycowork_file)
        graph = glycan.Glycan.from_iupac(sample)
        print(graph)
        print("#glycowords: %d\n" % len(graph.glycoword_type))
        torchglycan_file = os.path.join(output_dir, "unconnected_torchglycan_%d.pdf" % i)
        graph.visualize(save_file=torchglycan_file)
