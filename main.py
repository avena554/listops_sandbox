import torch
import torch.nn as nn
from models.tree_embedding_based_proba import TreeEmbeddingBasedProba
from models.christophs_tree_lstm import TreeLSTM
from models.n_ary_tree_lstm import NAryTreeLSTM, EmbeddingOnly, Debatched
from models.spinn import EmulatedBinaryTreeLSTM, Spinn
from inference_loops.tree_per_tree_loop import TreePerTreeLoop
from inference_loops.batch_per_batch_loop import BatchPerBatchLoop
from evaluation.eval import eval
from evaluation.list_ops_no_batch_accuracy import ListOpsNoBatchAccuracy
from list_ops_processing.listops import read_dataset
from inference_loops.point import Point
from test_ressources.test_strings import test_string_1
from nltk import Tree


device = 0
cuda_mode = True
use_cuda = torch.cuda.is_available() and cuda_mode
n_epochs = 15
lexicon_embedding_dim = 128
recursive_embedding_dim = 128
decoder_hdim = 128

#FIXME: compatibility with old non_batched models and train_loop is broken. Clean code and Restore it.
batch_mode = True

batch_size = 16
# switch to a tree instances minimal dataset for debugging purpose.
use_tiny_dataset = False

if use_cuda:
    def tree_to_device(tensor_tree):
        return Tree(tensor_tree.label().cuda(device),
                    [tree_to_device(child) for child in tensor_tree]
                    )

    def module_or_tensor_to_device(m):
        return m.cuda(device)
else:
    def tree_to_device(tensor_tree):
        return tensor_tree

    def module_or_tensor_to_device(m):
        return m



#class NoBatchLoss(nn.Module):
#    def __init__(self, loss: nn.Module):
#        super(NoBatchLoss, self).__init__()
#        self.actual_loss = loss

#    def forward(self, value, target : int):
#        one_batch_value = value.view(1, -1)
#        if value[0].item() != value[0].item():
#            print("overflowing")
#            exit(0)
#        tensor_target = module_or_tensor_to_device(torch.tensor([target]))
#        return self.actual_loss(one_batch_value, tensor_target)


def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def read_input_files(train_file, test_file):
    print("reading train")
    _trees_train, _interner = read_dataset(train_file)
    print("reading test")
    _trees_test, _ = read_dataset(test_file, _interner)
    "FIXME: we should have unknown happening here"
    print("done reading")
    return _trees_train, _trees_test, _interner


if use_cuda:
    str_device = "gpu %d" % device
else:
    str_device = "cpu"

print("Running on %s" % str_device)

if use_tiny_dataset:
    parts = test_string_1.splitlines()
    train_file, test_file = (parts, parts)
    trees_train, trees_test, interner = read_input_files(train_file, test_file)

else:
    with open("./listops-dataset/train_d20s.tsv") as train_file:
        with open("./listops-dataset/test_d20s.tsv") as test_file:
            trees_train, trees_test, interner = read_input_files(train_file, test_file)

data_points_train = [Point(point.tensor_tree, point.value) for point in trees_train]
data_points_test = [Point(tree_to_device(point.tensor_tree), point.value) for point in trees_test]
lexicon_size = interner.lexicon_size()
print("lexicon_size: %d" % lexicon_size)

if not batch_mode:
    #FIXME:broken
    #encoder = TreeLSTM(lexicon_size, 64, 64, 1, 0)
    #encoder = EmbeddingOnly(NAryTreeLSTM(2, module_or_tensor_to_device(torch.zeros(64, requires_grad=False)),
    #                                            lexicon_size, 64, 64))
    #training_loop_factory = TreePerTreeLoop
    #encoder = Debatched(EmbeddingOnly(EmulatedBinaryTreeLSTM(lexicon_size, 64)))
    pass
else:
    encoder = EmbeddingOnly(
        Spinn(lexicon_embedding_dim, recursive_embedding_dim, module_or_tensor_to_device(torch.zeros(recursive_embedding_dim, requires_grad=False)),
              lexicon_size, to_device=module_or_tensor_to_device)
    )
    training_loop_factory = BatchPerBatchLoop

    model = TreeEmbeddingBasedProba(10, recursive_embedding_dim, decoder_hdim, encoder)
    module_or_tensor_to_device(model)

    loss = nn.NLLLoss()
    #actual_loss = NoBatchLoss(base_loss)
    module_or_tensor_to_device(loss)

    loop = training_loop_factory(data_points_train, model, loss, module_or_tensor_to_device, lr=0.0001, batch_size=batch_size)

    score_accu = ListOpsNoBatchAccuracy(True)

    loop.train(n_epochs)

    print((eval(data_points_test, Debatched(EmulatedBinaryTreeLSTM(spinn=model)), score_accu)))
