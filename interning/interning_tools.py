import os


class IncreaseFunction:
    """
    This class is used to create sequence of increasing numbers
    to use in the interner. This means that every element is simply
    assigned the next number, with the starting number externally given
    but defaulting to 1.
    """

    def __init__(self, start=1):
        self.count = start

    def __call__(self, word):
        val = self.count
        self.count += 1
        return val

    def inform(self, maximum):
        self.count = maximum + 1


class Interner:
    """
    This class can be used as a interner for arbitrary objects.

    0 is the default key, all words that are not the default word must return a value greater than 0
    the semantics of the default word can be externally defined

    the not present function defines what numbers to assign to words that have not been seen
    by the interner before.

    no interned element may have a string representation that starts with two spaces
    this is intended to allow for comments in an initialization string.

    the initialization constructor can be used to transform words before they are added to the
    interner

    The string representation of this class is suitable as an initialzer for a new instance.

    The initialization string has the form:
    string to encode
    encoding number
    empty lines are ignored
      lines starting with two spaces are ignored
    """

    DEFAULT_INIT = []

    def __init__(self, not_present_function=None, initialization=None,
                 initialization_key_map=lambda x: x):
        if initialization is None:
            initialization = []
        if not_present_function is None:
            not_present_function = IncreaseFunction()

        self.entries = {}
        self.not_present_function = not_present_function
        self.max = 1

        self.reverse = {}

        seen = []
        for line in initialization:
            if line == "" or line.startswith('  '):
                continue
            else:
                seen.append(line.strip())
            if len(seen) >= 2:
                key = seen[0]
                value = int(seen[1])
                seen.clear()

                key = initialization_key_map(key)
                self.insert(value, key)

                if self.max < value:
                    self.max = value
        if self.max > 1:
            self.not_present_function.inform(self.max)
        self.expand = -1

    def __call__(self, word):
        key = self.entries.get(word, None)

        if key is None:
            if self.expand <= 0:
                key = self.not_present_function(word)
                self.insert(key, word)
                if key > self.max:
                    self.max = key
            else:
                return self.expand

        return key

    def insert(self, id_number, element):
        """

        :param id_number:
        :param element:
        :return:
        """
        self.entries[element] = id_number
        self.reverse[id_number] = element

        if id_number > self.max:
            self.max = id_number
            self.not_present_function.inform(self.max)

    def look_up(self, index: int):
        return self.reverse.get(index)

    def maximum(self):
        return self.max

    def lexicon_size(self):
        """
        Do not call this if additional symbols are still waiting to be interned.
        :return:
        """
        return self.max + 1

    def __repr__(self):
        info = []

        entries = self.entries.items()
        # entries = sorted(entries)
        for pair in entries:
            info.append(repr(pair[0]))
            info.append(repr(pair[1]))
        return os.linesep.join(info)

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return iter(self.entries.items())
