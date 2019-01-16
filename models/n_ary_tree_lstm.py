import torch
import torch.nn as nn
from nltk.tree import Tree


class NAryTreeLSTMCell(nn.Module):

    def __init__(self, arity, label_dim, h_dim):
        super(NAryTreeLSTMCell, self).__init__()

        self.label_dim = label_dim
        self.h_dim = h_dim
        self.arity = arity

        # dimension of a vector containing the node label's embedding and embeddings for the children
        self._arg_dim = label_dim + arity*h_dim
        # input gate linearity including components for the label, and all children + bias
        self.input_gate_lin = nn.Linear(self._arg_dim, self.h_dim)
        # output gate linearity, same as before
        self.output_gate_lin = nn.Linear(self._arg_dim, self.h_dim)
        # linearity for computing the update candidate, same as before
        self.update_lin = nn.Linear(self._arg_dim, self.h_dim)

        # Set up for the forget gate.
        # Each child has its own forget gate, computed using the node label and all its siblings (including itself)
        # We could make as many linear layers to be used in a for loop,
        # but instead we make a tensor of first dim arity with the appropriate weight matrices as elements.
        # to be used with the batched matrix multiplication of pytorch
        self.label_forget_weight_matrix = nn.Parameter(torch.zeros(h_dim, label_dim, requires_grad=True))
        self.siblings_forget_weight_matrices = nn.ParameterDict(
            dict(
                [
                    (
                        "(%d, %d)" % (k, l),
                        nn.Parameter(torch.zeros(self.h_dim, self.h_dim, requires_grad=True))
                    )
                    for k in range(self.arity) for l in range(self.arity)
                ]
            )
        )
        self.forget_bias = nn.Parameter(torch.zeros(self.h_dim, requires_grad=True))

        self.gates_nl = nn.Sigmoid()
        self.update_nl = nn.Tanh()

        self._init_forget_params()

    def _make_forget_matrices_batch(self, input_batch_size):
        all_forget_weights = [
            tuple(
                [self.label_forget_weight_matrix] +
                [self.siblings_forget_weight_matrices["(%d, %d)" % (k, l)]
                 for l in range(self.arity)]
            )
            for k in range(self.arity)
        ]
        concatenated_forget_weight_matrices = (
            torch.cat(forget_weights, dim=1) for forget_weights in all_forget_weights
        )
        # the input is batched as well, along a different dimension:
        # this will make us go from
        # arity, h_dim, _arg_dim to
        # arity, N, h_dim, _arg_dim with N the size of input minibatch
        batched_matrices = tuple(
            torch.stack(input_batch_size * [concatenated])
            for concatenated in concatenated_forget_weight_matrices
        )

        all_matrices = torch.stack(batched_matrices)

        return all_matrices.view(self.arity*input_batch_size, self.h_dim, self._arg_dim)

    def _init_forget_params(self):
        nn.init.xavier_uniform_(self.label_forget_weight_matrix)
        for _, p in self.siblings_forget_weight_matrices.items():
            nn.init.xavier_uniform_(p)

    def forward(self, label_and_children, children_cells_list):
        #get the size of the minibatch
        input_batch_size = label_and_children[0].size()[0]

        #print("Input vector: %s" % str(label_and_children))
        #print("Input cells: %s" % str(children_cells_list))

        label_and_children_vector = torch.cat(label_and_children, 1)

        children_cells = torch.stack(children_cells_list)

        input_gate_value = self.input_gate_lin(label_and_children_vector)
        input_gate_value = self.gates_nl(input_gate_value)

        #copy batched input vectors and view them as matrices with second dim 1.
        copied_input = torch.stack(self.arity*[label_and_children_vector.view(input_batch_size, self._arg_dim, 1)])
        #merge 2 first dims for bmm coming next
        copied_input = copied_input.view(self.arity*input_batch_size, self._arg_dim, 1)

        forget_matrices_batch = self._make_forget_matrices_batch(input_batch_size)

        #output will be input_batch_size*arity, h_dim, 1
        forget_gates_values = torch.bmm(forget_matrices_batch, copied_input)
        forget_gates_values = forget_gates_values.view(self.arity, input_batch_size, self.h_dim)
        forget_gates_values = self.gates_nl(forget_gates_values)

        output_gate_value = self.output_gate_lin(label_and_children_vector)
        output_gate_value = self.gates_nl(output_gate_value)

        update_candidate = self.update_lin(label_and_children_vector)
        update_candidate = self.update_nl(update_candidate)

        cells_retained = (forget_gates_values*children_cells)
        #print("retained: %s"%str(cells_retained.size()))

        cells_composed = cells_retained.sum(dim=0)

        cells_summed = children_cells.sum(dim=0)

        cells_forget_bias = self.forget_bias*cells_summed

        input_retained = input_gate_value*update_candidate

        next_cell = input_retained + cells_composed + cells_forget_bias

        out = output_gate_value*next_cell
        out = self.update_nl(out)

        #print("Out: %s"%str((next_cell, out)))

        return next_cell, out


class NAryTreeLSTM(nn.Module):
    def __init__(self, arity, leaf_cell_value, lexicon_size, label_dim, h_dim, encoder=None, tree_lstm_cell=None):
        super(NAryTreeLSTM, self).__init__()

        self.arity = arity
        self.leaf_cell_value = leaf_cell_value
        self.lexicon_size = lexicon_size
        self.label_dim = label_dim
        self.h_dim = h_dim

        if encoder:
            self.node_embedding_layer = encoder
        else:
            self.node_embedding_layer = nn.Embedding(lexicon_size, label_dim, padding_idx=0)

        if tree_lstm_cell:
            self.lstm_cell = tree_lstm_cell
        else:
            self.lstm_cell = NAryTreeLSTMCell(arity, label_dim, h_dim)

    def forward(self, tree:Tree):
        label = tree.label()

        label_embedding = self.node_embedding_layer(label).view(1, self.label_dim)
        children_tensor_pairs = [self(child) for child in tree]
        if children_tensor_pairs:
            children_cells = [child_cell for (child_cell, _) in children_tensor_pairs]
            children_embeddings = [child_embedding for (_, child_embedding) in children_tensor_pairs]
            #print("Out: %s" % str([c.size() for c in children_embeddings]))
        else:
            children_cells = self.arity*[self.leaf_cell_value.view(1, -1)]
            children_embeddings = self.arity*[self.leaf_cell_value.view(1, -1)]

        return self.lstm_cell([label_embedding]+children_embeddings, children_cells)


class EmbeddingOnly(nn.Module):
    def __init__(self, tree_lstm:NAryTreeLSTM):
        super(EmbeddingOnly, self).__init__()
        self.tree_lstm = tree_lstm

    def forward(self, tree):
        out = self.tree_lstm(tree)
        return out[1]

#TODO: move this to utilities file.
class Debatched(nn.Module):

    def __init__(self, emb:nn.Module):
        super(Debatched, self).__init__()
        self.emb = emb

    def forward(self, tree):
        out = self.emb(tree)
        return out.view(out.size()[1])