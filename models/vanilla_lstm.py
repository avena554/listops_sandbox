import torch
import torch.nn as nn

class VanillaLSTM(nn.Module):

    def __init__(self, lexicon_size, input_size, hidden_size, n_layers):
        super(VanillaLSTM, self).__init__()
        self.embedding = nn.Embedding(lexicon_size, input_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=0,
                            bidirectional=True)
        self.n_layers = n_layers
        self.h_dim = hidden_size

    def forward(self, seq):
        """

        :param seq: input tensor of longs of shape seq_length, batch_size
        :return: tensor of shape batch_size, 2*hidden_size
        """

        batch_size = seq.size()[1]

        lstm_in = self.embedding(seq)
        lstm_out = self.lstm(lstm_in)
        last_hidden = lstm_out[1][0]
        out = last_hidden.view(self.n_layers, 2, batch_size, -1)
        return out[-1].view(batch_size, 2*self.h_dim)
