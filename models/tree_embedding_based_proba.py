import torch.nn as nn


class TreeEmbeddingBasedProba(nn.Module):

    def __init__(self, n_outcomes: int, embedding_dim: int, decoder_hdim: int, tree_encoder: nn.Module):
        super(TreeEmbeddingBasedProba, self).__init__()
        self.tree_encoder = tree_encoder
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, decoder_hdim),
                                     nn.Tanh(),
                                     nn.Linear(decoder_hdim, n_outcomes),
                                     nn.Tanh())
        #self.decoder = nn.Linear(embedding_dim, n_outcomes)
        #self.nl = nn.Tanh()
        self.final_layer = nn.LogSoftmax(1)

    def forward(self, tree):
        embedding = self.tree_encoder(tree)
        logits = self.decoder(embedding)
        #logits = self.nl(logits)
        #pi/2 ~ 1.57
        logits = logits*1.57
        logits = logits.tan()

        probs = self.final_layer(logits)

        return probs
