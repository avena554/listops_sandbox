import unittest
from unittest import TestCase
from models.tree_embedding_based_proba import TreeEmbeddingBasedProba
from models.test_christophs_tree_lstm  import global_set_up
from models.christophs_tree_lstm import TreeLSTM


class TestTreeEmbeddingBasedProba(TestCase):

    def setUp(self):
        #makes pycharm static analysis happier
        self.data_point = None
        self.tree = None
        self.lexicon_size = None

        global_set_up(self)


    def test(self):
        encoder = TreeLSTM(self.lexicon_size, 8, 124, 1, 0)
        model = TreeEmbeddingBasedProba(10, 8, 124, encoder)
        print(model(self.tree))

