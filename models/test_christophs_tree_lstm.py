from unittest import TestCase
from test_ressources.tree_lstm_testing_utilities import global_set_up, test_forward
from models.christophs_tree_lstm import TreeLSTM


class TestChristophsTreeLSTM(TestCase):

    def setUp(self):
        #makes pycharm static analysis happier
        self.data_point = None
        self.tree = None
        self.lexicon_size = None

        global_set_up(self)
        self.model = TreeLSTM(self.lexicon_size, 8, 8, 1, 0)


    def test(self):
        test_forward(self.model, self.tree, 8)
