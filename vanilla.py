from unittest import TestCase
import torch
import torch.nn as nn
from models.tree_embedding_based_proba import TreeEmbeddingBasedProba
from models.vanilla_lstm import VanillaLSTM
from inference_loops.tree_per_tree_loop import TreePerTreeLoop
from evaluation.eval import eval
from evaluation.list_ops_no_batch_accuracy import ListOpsNoBatchAccuracy
from list_ops_processing.listops import read_dataset, ListOpsDataPoint
from test_ressources.test_strings import test_string_1


class Seq:
    def __init__(self, data_point:ListOpsDataPoint):
        self.tree = torch.tensor(data_point.word_ints).view(-1, 1)
        self.value = data_point.value

class LossForVanilla(nn.Module):
    def __init__(self):
        super(LossForVanilla, self).__init__()
        self.actual_loss = nn.NLLLoss()

    def forward(self, proba, value):
        return self.actual_loss.forward(proba.view(1, -1), torch.tensor([value]))

class Debatch(nn.Module):
    def __init__(self, other_module):
        super(Debatch, self).__init__()
        self.other_module = other_module

    def forward(self, *input):
        batched = self.other_module.forward(*input)
        return batched[0]


def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


#with open("./listops-dataset/first20train.tsv") as train_file:
#    with open("./listops-dataset/2130train.tsv") as test_file:
if(True):
        parts = test_string_1.splitlines()
        train_file, test_file = (parts, parts)
        print(train_file)
        print("reading train")
        trees_train, interner = read_dataset(train_file)
        print("reading test")
        trees_test, _ = read_dataset(test_file, interner)
        "FIXME: we should have unknown happening here"
        print("done reading")
        data_points_train = [Seq(point) for point in trees_train]
        data_points_test = [Seq(point) for point in trees_test]
        #print(trees_test[0].tree)
        #exit(0)
        lexicon_size = interner.lexicon_size()
        encoder = Debatch(VanillaLSTM(lexicon_size, 32, 32, 1))
        #print_params(encoder)
        model = TreeEmbeddingBasedProba(10, 64, 64, encoder)
        base_loss = LossForVanilla()
        loop = TreePerTreeLoop(data_points_train, model, base_loss)
        score_accu = ListOpsNoBatchAccuracy(True)

        loop.train(10)
        #print_params(encoder)
        print((eval(data_points_test, model, score_accu)))