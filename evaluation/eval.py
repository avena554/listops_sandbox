import torch.nn as nn
from evaluation.score_accumulator import ScoreAccumulator


def eval(test_data_points, model:nn.Module, score_accu:ScoreAccumulator) :
    for point in test_data_points:
        res = model.forward(point.tree)
        score_accu.accept(res, point.value)

    return score_accu.get_score()