import torch
from evaluation.score_accumulator import  ScoreAccumulator


class ListOpsNoBatchAccuracy(ScoreAccumulator):

    def __init__(self, verbose=False):
        super(ListOpsNoBatchAccuracy, self).__init__()
        self._n = 0
        self._acc = 0
        self._verbose = verbose

    def _indicator(self, thing, other):
        if thing == other:
            return 1
        else:
            return 0

    def accept(self, thing, other):
        self._n = self._n + 1
        argmax = torch.max(thing, 0)[1].item()
        self._acc = self._acc + self._indicator(argmax, other)
        if(self._verbose):
            print("gold value: %d, predicted value:%d"%(other, argmax))

    def get_score(self):
        return self._acc/self._n


