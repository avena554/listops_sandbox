from unittest import TestCase
import torch
import torch.nn as nn
from models.tree_embedding_based_proba import TreeEmbeddingBasedProba
from models.test_christophs_tree_lstm  import global_set_up
from models.christophs_tree_lstm import TreeLSTM
from inference_loops.tree_per_tree_loop import TreePerTreeLoop
from evaluation.eval import eval
from evaluation.list_ops_no_batch_accuracy import ListOpsNoBatchAccuracy


class TestTrainingTreeLSTM(TestCase):

    def setUp(self):
        #makes pycharm static analysis happier
        self.data_point = None
        self.tree = None
        self.lexicon_size = None

        global_set_up(self)
        self.value = self.data_point.value
        encoder = TreeLSTM(self.lexicon_size, 8, 124, 1, 0)
        self.model = TreeEmbeddingBasedProba(10, 8, 124, encoder)
        base_loss = nn.NLLLoss()
        actual_loss = NoBatchLoss(base_loss)
        self.loop = TreePerTreeLoop([self], self.model, actual_loss)
        self.score_accu = ListOpsNoBatchAccuracy(True)

    def test(self):
        self.loop.train(50)
        assert eval([self], self.model, self.score_accu) == 1


class NoBatchLoss(nn.Module):
    def __init__(self, loss: nn.Module):
        super(NoBatchLoss, self).__init__()
        self.actual_loss = loss

    def forward(self, value, target : int):
        one_batch_value = value.view(1, -1)
        tensor_target = torch.tensor([target])
        return self.actual_loss(one_batch_value, tensor_target)

