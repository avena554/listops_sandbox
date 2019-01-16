class CountNumberAssignment:
    """
    This function returns new numbers for an interner, but it assigns 1 to elements with a count
    below a given threshold. The counts and the threshold must be given externally.
    """

    def __init__(self, cut_off, counts, start=2):
        """

        :param cut_off: how often an element must have been seen to be assigned a number other than
        1
        :param counts: a dictionary from elements to count numbers
        :param start: the number at which we start, defaults to 2 (0 for padding and 1 for unknown words)
        """
        self.cut_off = cut_off
        self.counts = counts
        self.next_free_number = start

    def __call__(self, word):
        """
        Returns the interning number of the given word with 1 if the count is below the threshold.

        :param word:
        :return:
        """
        count = self.counts.get(word, 0)

        if count >= self.cut_off:
            val = self.next_free_number
            self.next_free_number += 1
            return val
        else:
            return 1

    def inform(self, maximum):
        self.next_free_number = maximum + 1


class NewIsOne:
    """
    This is essentially the default new word function, which maps each unknown
    word to 1
    """

    def __call__(self, word):
        return 1

    def inform(self, maximum):
        pass
