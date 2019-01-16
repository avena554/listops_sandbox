from unittest import TestCase
from test_ressources.tree_lstm_testing_utilities import global_set_up, test_forward
from util.shift_reduce_utils import binary_tree_to_sr, sr_to_tensor, as_batch
import torch
import torch.nn as nn
from models.spinn import Spinn
from models.n_ary_tree_lstm import EmbeddingOnly, NAryTreeLSTM


class TestSpinn(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # makes pycharm static analysis happier
        self.data_point = None
        self.tree = None
        self.lexicon_size = None

        global_set_up(self)
        #print(self.tree)
        self.sr_seq, self.label_seq = binary_tree_to_sr(self.tree)
        self.tensors = sr_to_tensor(self.sr_seq, self.label_seq)
        self.batch = as_batch([self.tensors])
        self.spinn = Spinn(16, 64, torch.zeros(64), self.lexicon_size)
        #print([p for p in self.label_encoder.named_parameters()])
        #print([p for p in self.spinn.named_parameters()])
        self.binaryTreeLSTM = EmbeddingOnly(NAryTreeLSTM(2, torch.zeros(64), self.lexicon_size, 16, 64, encoder=self.spinn.label_embedding, tree_lstm_cell=self.spinn.lstm))
        #print("*********************")
        #print([p for p in self.binaryTreeLSTM.named_parameters()])

    def test_as_batch(self):
        #print("starting spinn computation")
        out = self.spinn(self.batch)
        #print("starting regular computation")
        out_tree = self.binaryTreeLSTM(self.tree)
        #print("spinn: ",out[1])
        #print("tree lstm: ", out_tree)
        assert torch.all(out[1] == out_tree)

