from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer
from random import shuffle

class TreePerTreeLoop:
    def __init__(self, data_points, tree_model : Module, loss : Module, verbose: bool = True,
                 optimizer_factory : Optimizer = Adam, *optimizer_args, **optimizer_kwargs):
        self._data_points = list(data_points)
        self.data_points = data_points
        self.tree_model = tree_model
        self.loss = loss
        self.optimizer = optimizer_factory(self.tree_model.parameters(), *optimizer_args, **optimizer_kwargs)
        self.verbose = verbose


    def step(self, epoch = 0):
        cumulated_loss = 0.
        epoch_loss = 0.
        instances = self._get_data_iterable()
        for (i, instance) in instances:
            self.optimizer.zero_grad()

            value = self.tree_model(instance.tree)
            loss_and_grad = self.loss(value, instance.value)
            loss_and_grad.backward()

           # print(self.tree_model.tree_encoder.tree_lstm.lstm_cell)

            self.optimizer.step()

            loss_value = loss_and_grad.item()
            cumulated_loss += loss_value
            epoch_loss += loss_value

            if(self.verbose and i%1000 == 0):
                print("\t[epoch %d, instance %d] mean loss over the last 1000 examples: %f"%(epoch, i, cumulated_loss / 1000))
                cumulated_loss = 0.


        if(self.verbose):
            print("Epoch %d mean loss: %f"%(epoch, epoch_loss/i))

    def train(self, epochs : int):
        for epoch in range(1, epochs+1):
            self.step(epoch)

    def _get_data_iterable(self):
        shuffle(self._data_points)
        return enumerate(self._data_points, 1)
