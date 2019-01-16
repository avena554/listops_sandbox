from models.n_ary_tree_lstm import NAryTreeLSTMCell as TreeLSTMCell
import torch
import torch.nn as nn
from collections import deque
from nltk import Tree
from util.shift_reduce_utils import binary_tree_to_sr, sr_to_tensor, as_batch


def _get_indices(reduce_mask, active_pointers):
    indices = []
    for (i, q) in enumerate(active_pointers):
        if reduce_mask[i] == 0:
            index = -1
        else:
            index = q.pop()
        indices.append(index)

    return indices


class Spinn(nn.Module):

    def __init__(self, label_dim, h_dim, leaf_input, lexicon_size, encoder=None, tree_lstm_cell=None, to_device=lambda x:x):
        super(Spinn, self).__init__()
        self.label_dim = label_dim
        self.h_dim = h_dim
        self.lexicon_size = lexicon_size
        self._arg_dim = label_dim + 2*h_dim
        self.to_device = to_device
        if encoder:
            self.label_embedding = encoder
        else:
            self.label_embedding = nn.Embedding(lexicon_size, label_dim, padding_idx=0)
        if tree_lstm_cell:
            self.lstm = tree_lstm_cell
        else:
            self.lstm = TreeLSTMCell(2, label_dim, h_dim)
        self.leaf_input = leaf_input

    def _read(self, N, transitions, node_labels, thin_stack, step, active_pointers):
        embeddings_stack = thin_stack[0]
        cells_stack = thin_stack[1]

        reduce_mask = transitions[step]
        #1 for positions in the batch to be reduced. dim is N

        #get relevant labels. This is an (N, label_dim) tensor.
        labels = node_labels[step]

        #gather material for lstm call
        #get the two children embeddings and lstm cells from the stack, for all entries in the batch.
        args = {'left': [], 'right': []}
        cells = {'left': [], 'right': []}

        #print("preparing for reduce with %s" % str(thin_stack))
        for _dir in ('right', 'left'):
            #print("active pointers: %s" % str(active_pointers))
            step_indices = _get_indices(reduce_mask, active_pointers)
            for batch_pos, index in enumerate(step_indices):
                if index == -1:
                    args[_dir].append(self.leaf_input)
                    cells[_dir].append(self.leaf_input)

                else:
                    args[_dir].append(embeddings_stack[index, batch_pos])
                    cells[_dir].append(cells_stack[index, batch_pos])

            #print(_dir, [x.size() for x in args[_dir]])


        #stack the lists into N, h_dim tensors
        left_arg_batched = torch.stack(args['left'])
        right_arg_batched = torch.stack(args['right'])

        left_cell_batched = torch.stack(cells['left'])
        right_cell_batched = torch.stack(cells['right'])

        #final arguments for calling the tree lstm cell
        label_and_children = (labels, left_arg_batched, right_arg_batched)
        children_cells = (left_cell_batched, right_cell_batched)

        #call the lstm.
        reduce_out = self.lstm(label_and_children, children_cells)

        #update the thin_matrix
        embeddings_stack[step] = reduce_out[1]
        cells_stack[step] = reduce_out[0]

        #add current step to active pointers everywhere
        for i in range(N):
            active_pointers[i].append(step)

    def forward(self, tree_as_sr):
        """

        :param tree_as_sr: pair, first component of which is a
         D,N tensor representing the transitions (1 for reduce, 0 for shift)
        and second component of which is a D, N, tensor representing
        the index (in the lexicon) of the node labels used by corresponding
        shift or reduce operations.
        """
        (transitions, node_labels_indices) = tree_as_sr
        #Size of batch
        N = len(transitions[0])
        #Number of transitions
        D = len(transitions)

        # get embeddings for all labels
        node_labels = self.label_embedding(node_labels_indices)

        #Data structure to efficiently implement the batched stacks of the Spinn network.
        thin_stack = self.to_device(torch.zeros(2, D, N, self.h_dim, requires_grad=False))
        #pointers to entries of the thin stack which are still active
        active_pointers = [deque() for _ in range(N)]

        for step in range(D):
            self._read(N, transitions, node_labels, thin_stack, step, active_pointers)

        out = thin_stack[1][D - 1], thin_stack[0][D - 1]
        return out


class EmulatedBinaryTreeLSTM(nn.Module):

    def __init__(self, h_dim=None, spinn=None):
        super(EmulatedBinaryTreeLSTM, self).__init__()
        if spinn:
            self.spinn = spinn
        else:
            self.spinn = Spinn(h_dim, h_dim, torch.zeros(h_dim), self.lexicon_size)

    def forward(self, tree:Tree):
        sr_seq, lbl_seq = binary_tree_to_sr(tree)
        tensors = sr_to_tensor(sr_seq,lbl_seq)
        batched = as_batch([tensors])

        out = self.spinn(batched)
        return out





