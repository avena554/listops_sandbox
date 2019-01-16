from unittest import TestCase

import nltk.tree as tree_mod
import torch

import list_ops_processing.listops as listops
from test_ressources.test_strings import test_string_1 as test_string

"""test_string = '6	( ( ( ( ( [MAX ( ( ( ( ( ( [MED 4 ) 6 ) 6 ) 0 ) 8 ) ] ) ) 6 ) 6 ) 0 ) ] )\n 7	( ( ( ( [SM ( ( ( ' \
              '[MED 6 ) 5 ) ] ) ) 1 ) 1 ) ] )\n 4	( ( ( ( ( [MAX 3 ) 4 ) 3 ) 3 ) ] )\n '
"""

class TestListOpsDataPoint(TestCase):

    def setUp(self):
        parts = test_string.splitlines()

        self.trees, self.interner = listops.read_dataset(parts)
        print(self.trees)

    def test_data(self):
        self.assertEqual(len(self.trees), 3)

        self.assertEqual(self.interner("X"), 10)
        self.assertEqual(self.interner("[MAX"), 11)
        self.assertEqual(self.interner("3"), 3)
        self.assertEqual(self.interner("0"), 0)
        self.assertEqual(self.interner("13"), 15)
        self.assertEqual(self.interner("]"), 13)

        t = self.trees[2]

        self.assertEqual(t.value, 4)

        self.assertTrue(isinstance(t.tree, tree_mod.Tree))
        self.assertEqual(t.tree, tree_mod.Tree.fromstring("(X (X (X (X (X ([MAX ) (3 )) (4 )) (3 )) (3 )) (] ))"))

        self.assertTrue(isinstance(t.int_tree, tree_mod.Tree))
        self.assertEqual(t.int_tree[0].label(), 10)
        self.assertEqual(len(t.int_tree[0][0][0][0]), 2)
        self.assertEqual(t.int_tree[0][0][0][0][0].label(), 11)

        self.assertEqual(t.words, ["[MAX", "3", "4", "3", "3", "]"])
        self.assertEqual(t.word_ints, [11, 3, 4, 3, 3, 13])

        self.assertTrue(isinstance(t.tensor_tree[0].label(), torch.LongTensor))
        self.assertEqual(len(t.tensor_tree.label()), 1)
        self.assertEqual(t.tensor_tree.label().item(), 10)
