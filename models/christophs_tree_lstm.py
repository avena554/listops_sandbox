import torch
import torch.nn as nn
from nltk.tree import Tree

class TreeLSTM(nn.Module):

    def __init__(self, node_lexicon_size: int, node_embedding_dimension: int,
                 lstm_dimension: int, lstm_layers: int, dropout: float):
        super(TreeLSTM, self).__init__()
        self.node_look_up = nn.Embedding(node_lexicon_size, node_embedding_dimension, padding_idx=0)
        self.node_lstm = nn.LSTM(input_size=node_embedding_dimension,
                                 num_layers=lstm_layers, hidden_size=lstm_dimension, dropout=dropout,
                                 bidirectional=True)

        self.num_layers = lstm_layers

        self.final_linearity = nn.Linear(in_features=2 * lstm_dimension, out_features=node_embedding_dimension)
        self.non_linearity = nn.Tanh()
        #self.non_linearity = nn.ReLU()

    def forward(self, tree: Tree):
        index = tree.label()

        # look up will give us a matrix, but that is hard to concat with the vectors coming from
        # the children, so we turn it into a vector...
        node = self.node_look_up(index)[0]

        data = node
        # concat it with the info from the children into a single vector ...
        for child in tree:
            data = torch.cat((data, self.forward(child)))

        # append the node embedding again to make sure backwards pass has enough info
        data = torch.cat((data, node))

        # and then resize the concatenation so we actually get a sequence of the correct length
        # (+2 since len gives us the number of children and we added the actual node label embedding
        # twice)
        data = data.view(len(tree) + 2, 1, -1)

        # should no longer need initial state - defaults to 0 in newer pytorch versions
        # index gets us the hidden state end of sequence vector
        # this will have dimensions: (num_layers * num_directions, batch, hidden_size)
        # we assume that there is exactly one element in the batch and we only want the last layer (?)
        embedding = self.node_lstm(data)

        encoded = embedding[1][0]
        # so we will re-size to (num_layers,num_directions, all the rest):
        encoded = encoded.view(self.num_layers, 2, -1)
        # and then retrieve only the last layer and merge the forward and backward direction into a single
        # vector:
        encoded = encoded[-1].view(-1)

        # then we squash the whole through a FF layer for the next level in the tree or the output:
        encoded = self.final_linearity(encoded)
        return self.non_linearity(encoded)
