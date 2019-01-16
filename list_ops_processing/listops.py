import csv
import os
import sys

import nltk.tree as tree_mod
import torch

import interning.interning_tools as interners

import re


ws_cleaner = re.compile("[\s]+")
def clean_ws(s):
    return ws_cleaner.sub('', s)

class ListOpsDataPoint:
    """"
    Instances of this class represent an training example from the list ops data set. All relevant informations and
    transformations of information are accessible through fields.

    The fields are:

    self.value -- value that the expression from the example evaluates to
    self.tree -- gold parse (nltk.tree.)Tree assigned to the example in the data set
    self.int_tree -- tree with the node labels replaced by ints which correspond to the interner from the instance
        construction
    self.words -- the sequence of words (operations and numbers) from the instance
    self.word_ints -- the sequence of numbers which correspond to the interned words of the instance
    self.tensor_tree -- the int tree with the ints replaced by torch.LongTensor(s) if dimensionality [1] that contain
        the int value.

    """

    def __init__(self, value_string, expression_string, interner: interners.Interner):
        self.value = int(value_string.strip())
        self.tree = all_nodes(tree_mod.Tree.fromstring(preprocess(expression_string.strip())))
        self.int_tree = to_numbers(self.tree, interner)
        self.words = get_yield(self.tree)
        self.word_ints = [interner(x) for x in self.words]
        self.tensor_tree = to_tensor(self.int_tree)

    def __repr__(self):
        parts = []
        parts.append("value: " + repr(self.value))
        parts.append("tree: " + " ".join([x.strip() for x in str(self.tree).strip().splitlines()]))
        parts.append("int tree: " + " ".join([x.strip() for x in str(self.int_tree).strip().splitlines()]))
        parts.append("words: " + " ".join(self.words))
        parts.append("word ints: " + " ".join([repr(y) for y in self.word_ints]))

        return os.linesep.join(parts)

class ExpressionDataPoint():
    def __init__(self, decorated:ListOpsDataPoint, interner: interners.Interner):
        self.value = decorated.value
        self.tree, _ = percolate(decorated.tree)
        self.int_tree = to_numbers(self.tree, interner)
        self.words = decorated.words
        self.word_ints = decorated.word_ints
        self.tensor_tree = to_tensor(self.int_tree)

def percolate(tree):
    pass

def read_dataset(file, interner=None, point_factory=ListOpsDataPoint):
    """
    This method reads a file (actually jsut an object for which __next_item__() returns a string) in the format of
    the list-ops dataset and returns a data set consisting of ListOpDataPoint(s) it also returns an interner
    (see interning.interning_tools.Interner) which maps node labels to numbers and vice versa.

    :param file:
    :param interner:
    :param point_factory:
    :return:
    """
    data_set = []
    r = range(0, 10)

    if (interner == None):
        func = RespectfulNumbering(respect=r)
        interner = interners.Interner(not_present_function=func)


    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        #print(row)
        if len(row) > 1:
            try:
                data_set.append(point_factory(row[0], row[1], interner))
            except ValueError:
                print("Warning, could not read line: %s"%row, file=sys.stderr)

    return data_set, interner


def preprocess(expression_string: str):
    """
    Replaces all '(' occurrences with '(X' so nodes have a non-empty label.

    :param expression_string:
    :return:
    """
    return expression_string.replace("(", "(X")


def get_yield(node):
    """
    Takes a tree in which all nodes are trees and returns a list of the labels of all nodes which have no children
    in left to right order.

    :param node:
    :return:
    """
    if len(node) > 0:
        l = []
        for child in node:
            l.extend(get_yield(child))
        return l
    else:
        return [node.label()]


def to_numbers(tree: tree_mod.Tree, interner: interners.Interner):
    """
    Maps a tree to a tree of int(s) which are the numbers assigned to the node labels by the interner.

    :param tree:
    :param interner:
    :return:
    """
    num = interner(tree.label().strip())

    return tree_mod.Tree(num, [to_numbers(t, interner) for t in tree])


def to_tensor(tree: tree_mod.Tree):
    """
    Maps a tree of int(s) to a tree of torch.LongTensor(s) which contain the same values.

    :param tree:
    :return:
    """
    lab = torch.LongTensor([int(tree.label())])
    children = [to_tensor(t) for t in tree]

    return tree_mod.Tree(lab, children)


def all_nodes(tree: tree_mod.Tree):
    """
    Ensures that a tree contains only nodes that are also trees. This means that leaf nodes that are

    :param tree:
    :return:
    """
    if isinstance(tree, tree_mod.Tree):
        return tree_mod.Tree(tree.label(), [all_nodes(child) for child in tree])
    else:
        return tree_mod.Tree(tree, [])


class RespectfulNumbering:
    """
    Tells an interner to map strings corresponding to initial number assignments for an interner. Strings corresponding
    to digits in a range that needs to be respected (given at construction) are mapped to the corresponding int. All
    other strings are mapped to int outside of the range.
    """

    def __init__(self, respect=range(0, 10)):
        self.respect = respect
        self.count = max(self.respect) + 1

    def __call__(self, word):
        if isinstance(word, str) and word.isdigit():
            num = int(word)
            if num in self.respect:
                return num

        val = self.count
        self.count += 1
        return val

    def inform(self, maximum):
        self.count = maximum + 1


