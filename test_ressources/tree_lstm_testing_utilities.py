from unittest import TestCase
from test_ressources.test_strings import single_tree_string
from list_ops_processing.listops import read_dataset
from models.christophs_tree_lstm import TreeLSTM
import torch


def global_set_up(object):
    trees, interner = read_dataset([single_tree_string])
    object.data_point = trees[0]
    object.tree = object.data_point.tensor_tree
    object.lexicon_size = interner.lexicon_size()


def test_forward(tree_lstm, tree, dim, batched=False, batch_size=1):
    embedding = tree_lstm(tree)
    #print(embedding)
    #print(embedding.size())

    if batched:
        expected = torch.Size([batch_size, dim])
    else:
        expected = torch.Size([dim])

    assert(embedding.size() == expected)
