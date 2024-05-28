import os
import sys
import argparse
import torch
from matplotlib import pyplot as plt

from glycowork.glycan_data.loader import *
from glycowork.glycan_data.data_entry import *

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import module.custom_models.models as custom_models
from module.custom_data.glycan import Glycan, PackedGlycan


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model selection", required=True)
    parser.add_argument("--batch_size", type=int, default=10, help="number of glycans in each batch")
    args = parser.parse_known_args()[0]

    return args


if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    half_batch_size = batch_size // 2
    model_name = args.model

    all_samples = df_species["glycan"]
    connected_samples = [sample for sample in all_samples if "{" not in sample and "}" not in sample]
    unconnected_samples = [sample for sample in all_samples if "{" in sample or "}" in sample]
    iupacs = connected_samples[:half_batch_size] + unconnected_samples[:half_batch_size]
    graphs = PackedGlycan.from_iupac(iupacs)
    print(graphs)
    print("Batch size:", half_batch_size * 2)
    print("Glycoword number:", graphs.num_glycowords, graphs.num_glycoword)
    print("Glycoword type shape:", graphs.glycoword_type.shape)
    print("Glycoword type:", graphs.glycoword_type)
    print("")

    input_dim = 3       # embedding, manually set
    num_glycoword_type = len(graphs.glycowords)     # number of types of glycowords
    if model_name == "GlycanCNN":
        model = custom_models.GlycanConvolutionalNetwork(input_dim=input_dim, hidden_dims=[256, 256], glycoword_dim=num_glycoword_type)
    elif model_name == "GlycanResNet":
        model = custom_models.GlycanResNet(input_dim=input_dim, hidden_dims=[256, 256], glycoword_dim=num_glycoword_type)
    elif model_name == "GlycanLSTM":
        model = custom_models.GlycanLSTM(input_dim=input_dim, hidden_dim=256, glycoword_dim=num_glycoword_type, num_layers=2)
    elif model_name == "GlycanBERT":
        # no self embedding
        model = custom_models.GlycanBERT(input_dim=num_glycoword_type)
    else:
        print("Available models: GlycanCNN, GlycanResNet, GlycanLSTM, GlycanBERT")
    out = model.forward(graphs, None)
    print(out, out["glycoword_feature"].shape, out["graph_feature"].shape, sep='\n')
