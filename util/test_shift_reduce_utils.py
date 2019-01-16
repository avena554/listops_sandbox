from unittest import TestCase
from test_ressources.tree_lstm_testing_utilities import global_set_up, test_forward
from util.shift_reduce_utils import binary_tree_to_sr, sr_to_binary_tree, sr_to_tensor, as_batch
import torch.nn as nn


class TestBinaryToSR(TestCase):
    def setUp(self):
        # makes pycharm static analysis happier
        self.data_point = None
        self.tree = None
        self.lexicon_size = None

        global_set_up(self)

    def test_converstion(self):
        seqs = binary_tree_to_sr(self.tree)
        back = sr_to_binary_tree(*seqs)
        assert back == self.tree


class TestSRToTensor(TestCase):
    def setUp(self):
        # makes pycharm static analysis happier
        self.data_point = None
        self.tree = None
        self.lexicon_size = None

        global_set_up(self)
        #self.label_encoder = nn.Embedding(self.lexicon_size, 16, padding_idx=0)
        self.sr_seq, self.label_seq = binary_tree_to_sr(self.tree)

    def test_sr_to_tensor(self):
        tensors = sr_to_tensor(self.sr_seq, self.label_seq)
        print(tensors)
        print(tensors[0].size(), tensors[1].size())


class TestSRBatch(TestCase):
    def setUp(self):
        # makes pycharm static analysis happier
        self.data_point = None
        self.tree = None
        self.lexicon_size = None

        global_set_up(self)
        self.label_encoder = nn.Embedding(self.lexicon_size, 16, padding_idx=0)
        self.sr_seq, self.label_seq = binary_tree_to_sr(self.tree)
        self.tensors = sr_to_tensor(self.sr_seq, self.label_seq)
        self.batch = [self.tensors, self.tensors, self.tensors]

    def test_as_batch(self):
        actual_batch = as_batch(self.batch)
        embedded = actual_batch[0], self.label_encoder(actual_batch[1])
        print(embedded[1].size())
