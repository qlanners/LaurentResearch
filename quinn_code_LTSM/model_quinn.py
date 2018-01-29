import torch
from torch.autograd import Variable
import torch.nn as nn
import cell_quinn


class RNNModel(nn.Module):

    def __init__(self, num_token, embed_size, hidden_size, dropout, tie_weights):
        super(RNNModel, self).__init__()

        self.drop = nn.Dropout(0.65)
        self.encoder = nn.Embedding(num_token, embed_size)
        self.rnn = cell_quinn.LTSMcell(embed_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_token) #why not have bias=True?

        if tie_weights:
            if embed_size != hidden_size:
                raise ValueError('When using the tied flag, embedding_size must be equal to hidden_size')
            self.decoder.weight = self.encoder.weight

        self.hidden_size=hidden_size    
        self.init_weights_encoder_decoder()

    def init_weights_encoder_decoder(self):
        initrange = 0.07
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.uniform_(-initrange, initrange)

    def forward(self, x, h_prev, c_prev):
        embed = self.encoder(x)
        h,c = self.rnn(embed, (h_prev,c_prev))
        h_dropped = self.drop(h)
        scores = self.decoder(h_dropped)
        return scores, h, c