from nltk import Tree
import torch


def binary_tree_to_sr(tree):
    children = [child for child in tree]
    if not children:
        return ["s"], [tree.label()]
    else:
        trs_and_lbls = [binary_tree_to_sr(child) for child in children]
        return (trs_and_lbls[0][0] + trs_and_lbls[1][0] + ["r"]), \
               (trs_and_lbls[0][1] + trs_and_lbls[1][1] + [tree.label()])


def sr_to_binary_tree(sr_seq, labels_seq):
    remaining = list(sr_seq)
    remaining.reverse()
    remaining_labels = list(labels_seq)
    remaining_labels.reverse()
    stack = []

    while remaining:
        trans = remaining.pop()
        if trans == 's':
            stack.append(Tree(remaining_labels.pop(), []))

        else:
            right_child = stack.pop()
            left_child = stack.pop()
            stack.append(Tree(remaining_labels.pop(), [left_child, right_child]))

    return stack.pop()


def sr_to_tensor(sr_seq, labels_seq, to_device=lambda x:x):
    sr_tensor = to_device(torch.FloatTensor([trs == 'r' for trs in sr_seq]))
    labels_embs_seq = to_device(torch.stack([lbl for lbl in labels_seq]))
    return sr_tensor, labels_embs_seq


def as_batch(sr_tensors):
    sr_seqs = [sr_tensor for (sr_tensor, _) in sr_tensors]
    labels_embs = [labels_tensor for (_, labels_tensor) in sr_tensors]

    sr_batch = torch.stack(sr_seqs).transpose(0, 1)
    labels_batch = torch.cat(labels_embs, 1)

    return sr_batch, labels_batch

