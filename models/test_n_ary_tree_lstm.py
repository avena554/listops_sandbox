from unittest import TestCase
from models.n_ary_tree_lstm import NAryTreeLSTMCell, NAryTreeLSTM, EmbeddingOnly
import torch
import torch.nn as nn
from unittest import TestCase
from test_ressources.tree_lstm_testing_utilities import global_set_up, test_forward


class TestNAryCell(TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.lstm = NAryTreeLSTMCell(3, 2, 3)
        label = torch.zeros(2).view(1,2)
        child1 = torch.zeros(3).view(1,3)
        child2 = torch.zeros(3).view(1,3)
        child3 = torch.zeros(3).view(1,3)

        cell1 = torch.zeros(3).view(1,3)
        cell2 = torch.zeros(3).view(1,3)
        cell3 = torch.zeros(3).view(1,3)

        self.label_and_children = (label, child1, child2, child3)
        self.cell = (cell1, cell2, cell3)

    def test(self):
        print(self.lstm(self.label_and_children, self.cell))


class TestNaryTreeLSTM(TestCase):

    def setUp(self):
        #makes pycharm static analysis happier
        self.data_point = None
        self.tree = None
        self.lexicon_size = None

        global_set_up(self)
        self.model = EmbeddingOnly(NAryTreeLSTM(2, torch.zeros(64), self.lexicon_size, 8, 64))

    def test(self):
        test_forward(self.model, self.tree, 64, batched=True)