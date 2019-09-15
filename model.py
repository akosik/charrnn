import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CharRnn(nn.Module):
    def __init__(self, rnn_type, ninpu, nhidu, ntoks, nlayers, allowed_tokens=string.printable, dropout=0.5):
        super(CharRnn, self).__init__()

        self.char2id = {c: i for i,c in enumerate(string.printable)}
        self.ninpu = ninpu
        self.nhidu = nhidu
        self.ntoks = ntoks
        self.nlayers = nlayers
        self.rnn_type = rnn_type

        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(ntoks, nhidu)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(nhidu, nhidu, nlayers, dropout=dropout)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(nhidu, nhidu, nlayers, dropout=dropout, nonlinearity='relu')

        self.decoder = nn.Linear(nhidu, ntoks)

    def forward(self, x, rnn_hidden):
        emb = self.dropout(self.emb(x))
        emb = emb.squeeze(2).permute(1, 0, 2)
        rnn_out, rnn_hidden = self.rnn(emb, rnn_hidden)
        output = self.decoder(rnn_out.permute(1, 0, 2))
        return output, rnn_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhidu),
                    weight.new_zeros(self.nlayers, bsz, self.nhidu))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
