import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, dropout = dropout)
        self.out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        attn_weights = self.attention(hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        context = context.squeeze(1)
        embedded_context = torch.cat([embedded, context.unsqueeze(0)], 2)
        output, (hidden, cell) = self.rnn(embedded_context, (hidden, cell))
        output = output.squeeze(0)

        context = context.squeeze(1)
        output = self.out(torch.cat([output, context], 1))
        print(output.shape)
        print(context.shape)
        return output, (hidden, cell)
