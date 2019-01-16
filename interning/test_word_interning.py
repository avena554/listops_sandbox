import os
import unittest

from interning import interning_tools


class AlmostCounter:

    def __init__(self):
        self.count = 1

    def __call__(self, word):
        if word == 'b':
            return 1
        else:
            self.count += 1
            return self.count

    def inform(self, x):
        pass


target_string = """'a'
2
'b'
1
'c'
3
'd'
4"""


class TestInterning(unittest.TestCase):

    def setUp(self):
        self.words = ["a", "b", "c", "a", "b", "c", "d"]
        self.func = AlmostCounter()
        self.interner = interning_tools.Interner(self.func)

    def test_interning(self):
        results = []
        for word in self.words:
            results.append(self.interner(word))

        self.assertEqual(results, [2, 1, 3, 2, 1, 3, 4])

        srep = self.interner.__repr__()
        self.assertEqual(srep, target_string)

        less = lambda x: x[1:len(x) - 1]

        other = interning_tools.Interner(initialization=srep.splitlines(), initialization_key_map=less)
        other("g")
        srep = other.__repr__()
        self.assertEqual(srep, target_string + os.linesep.join(("", "'g'", "5")))


class TestIncreaseFunction(unittest.TestCase):

    def test(self):
        func = interning_tools.IncreaseFunction(start=3)
        self.assertEqual(3, func("a"))
        self.assertEqual(4, func("a"))

        func.inform(19)
        self.assertEqual(20, func("a"))
        self.assertEqual(21, func("a"))
