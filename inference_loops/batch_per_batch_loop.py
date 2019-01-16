import torch
from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer
from util.shift_reduce_utils import binary_tree_to_sr, sr_to_tensor, as_batch
from random import shuffle
from inference_loops.point import Point


def _make_batch(data_slice, to_device):
    values = [to_device(torch.tensor([value])) for (value, _) in data_slice]
    values_batch = torch.cat(values)
    seqs = [seqs for (_, seqs) in data_slice]

    max_len = max([len(s[0]) for s in seqs])
    #padding with shift. The corresponding label should never affect gradient so should not matter
    #So we take 0.
    padding_label = torch.LongTensor([0])
    padded_seqs = [(((max_len-len(s[0]))*['s'] + s[0]), (max_len-len(s[0]))*[padding_label] + s[1]) for s in seqs]

    tensors = [sr_to_tensor(*s, to_device) for s in padded_seqs]
    batch = as_batch(tensors)

    return Point(batch, values_batch)


class BatchPerBatchLoop:
    def __init__(self, data_points, tree_model: Module, loss: Module, to_device, verbose: bool=True, batch_size=10,
                 optimizer_factory: Optimizer = Adam, *optimizer_args, **optimizer_kwargs):
        self.data_points = data_points
        self._data_points = [(point.value, binary_tree_to_sr(point.tree)) for point in self.data_points]
        self.n_data = len(self.data_points)
        self.tree_model = tree_model
        self.loss = loss
        self.to_device = to_device
        self.optimizer = optimizer_factory(self.tree_model.parameters(), *optimizer_args, **optimizer_kwargs)
        self.batch_size = batch_size
        self.verbose = verbose

    def step(self, epoch=0):
        cumulated_loss = 0.
        epoch_loss = 0.
        instances = self._get_data_iterable(self.batch_size)
        for (i, instance) in instances:
            self.optimizer.zero_grad()

            value = self.tree_model(instance.tree)
            loss_and_grad = self.loss(value, instance.value)
            loss_and_grad.backward()

            self.optimizer.step()

            loss_value = loss_and_grad.item()
            cumulated_loss += loss_value
            epoch_loss += loss_value

            if self.verbose and i % 10 == 0:
                print("\t[epoch %d, batch number %d] mean loss over the last 10 batches: %f" % (epoch, i, cumulated_loss / 10))
                cumulated_loss = 0.

        if self.verbose:
            print("Epoch %d mean loss: %f" % (epoch, epoch_loss/i))

    def train(self, epochs : int):
        for epoch in range(1, epochs+1):
            self.step(epoch)

    def _get_data_iterable(self, N):
        shuffle(self._data_points)
        n_batches = len(self._data_points) // N + (len(self._data_points) > 1)
        for i in range(n_batches):
            if i*N < self.n_data:
                yield (i+1, _make_batch(self._data_points[i*N:(i+1)*N], self.to_device))


