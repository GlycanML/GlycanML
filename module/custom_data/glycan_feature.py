import re
import os
import warnings
import pickle as pkl
from glycowork.motif import tokenization

from torchdrug.core import Registry as R

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "glycoword_vocab.pkl")
entity_vocab = pkl.load(open(data_path, "rb"))
unit_vocab = [entity for entity in entity_vocab
              if not (entity.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", entity))]
link_vocab = [entity for entity in entity_vocab
              if entity.startswith(("a", "b", "?")) or re.match("^[0-9]+(-[0-9]+)+$", entity)]
unit2category = {'ManHep': 'Hep', 'GalHep': 'Hep', 'DDGlcHep': 'Hep', 'DLGlcHep': 'Hep', 'IdoHep': 'Hep', 'LDManHep': 'Hep', 
                 'DDManHep': 'Hep', 'LyxHep': 'Hep', 'DDAltHep': 'Hep', 'Man': 'Hex', 'Glc': 'Hex', 'Galf': 'Hex', 'Hex': 'Hex', 
                 'Ido': 'Hex', 'Tal': 'Hex', 'Gul': 'Hex', 'All': 'Hex', 'Manf': 'Hex', 'Alt': 'Hex', 'Gal': 'Hex', 'Ins': 'Hex', 
                 'GulA': 'HexA', 'GalA': 'HexA', 'AltA': 'HexA', 'TalA': 'HexA', 'AllA': 'HexA', 'IdoA': 'HexA', 'GlcA': 'HexA', 
                 'HexA': 'HexA', 'ManA': 'HexA', 'ManN': 'HexN', 'AltN': 'HexN', 'GalN': 'HexN', 'HexN': 'HexN', 'AllN': 'HexN', 
                 'TalN': 'HexN', 'GulN': 'HexN', 'GlcN': 'HexN', 'GlcNAc': 'HexNAc', 'ManNAc': 'HexNAc', 'IdoNAc': 'HexNAc', 
                 'GlcfNAc': 'HexNAc', 'AltNAc': 'HexNAc', 'TalNAc': 'HexNAc', 'ManfNAc': 'HexNAc', 'GulNAc': 'HexNAc', 
                 'GalNAc': 'HexNAc', 'GalfNAc': 'HexNAc', 'AllNAc': 'HexNAc', 'HexNAc': 'HexNAc', 'Sor': 'Ket', 'Tag': 'Ket', 
                 'Fruf': 'Ket', 'Psi': 'Ket', 'Sedf': 'Ket', 'Fru': 'Ket', 'Xluf': 'Ket', 'Mur': 'Others', 'Erwiniose': 'Others', 
                 'MurNAc': 'Others', 'Pse': 'Others', 'Dha': 'Others', 'Fus': 'Others', 'Ko': 'Others', 'Pau': 'Others', 
                 'Aco': 'Others', 'IdoNGlcf': 'Others', 'dNon': 'Others', 'MurNGc': 'Others', 'ddNon': 'Others', 'Aci': 'Others', 
                 'Leg': 'Others', 'AcoNAc': 'Others', 'Api': 'Others', 'Apif': 'Others', 'Kdof': 'Others', 'Bac': 'Others', 
                 'Kdo': 'Others', 'Yer': 'Others', '4eLeg': 'Others', 'Ribf': 'Pen', 'Rib': 'Pen', 'Xyl': 'Pen', 'Ara': 'Pen', 
                 'Araf': 'Pen', 'Lyxf': 'Pen', 'Pen': 'Pen', 'Lyx': 'Pen', 'Xylf': 'Pen', 'AraN': 'PenN', 'Sia': 'Sia', 
                 'Neu5Ac': 'Sia', 'Neu5Gc': 'Sia', 'Neu': 'Sia', 'Kdn': 'Sia', 'Neu4Ac': 'Sia', 'Thre-ol': 'Tetol', 'Ery-ol': 'Tetol', 
                 '6dAltf': 'dHex', 'dHex': 'dHex', 'Qui': 'dHex', 'Rha': 'dHex', 'RhaN': 'dHex', 'Fuc': 'dHex', '6dAlt': 'dHex', 
                 'Fucf': 'dHex', '6dGul': 'dHex', '6dTal': 'dHex', 'QuiN': 'dHexN', 'FucN': 'dHexN', 'QuiNAc': 'dHexNAc', 
                 '6dTalNAc': 'dHexNAc', '6dAltNAc': 'dHexNAc', 'dHexNAc': 'dHexNAc', 'RhaNAc': 'dHexNAc', 'FucNAc': 'dHexNAc', 
                 'FucfNAc': 'dHexNAc', 'Par': 'ddHex', 'Asc': 'ddHex', 'Col': 'ddHex', 'Tyv': 'ddHex', 'Dig': 'ddHex', 'Oli': 'ddHex', 
                 'ddHex': 'ddHex', 'Abe': 'ddHex'}
category_vocab = list(set(unit2category.values()))


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature


@R.register("features.unit.symbol")
def unit_symbol(unit):
    """
    Symbol unit feature.
    """
    return onehot(tokenization.get_core(unit), unit_vocab)


@R.register("features.unit.default")
def unit_default(unit):
    """
    Default unit feature.
    """
    category = unit2category.get(tokenization.get_core(unit), "unknown_category")
    return unit_symbol(unit) + \
           onehot(category, category_vocab, allow_unknown=True)


@R.register("features.link.default")
def link_default(link):
    """
    Default link feature.
    """
    return onehot(tokenization.get_core(link), link_vocab)
