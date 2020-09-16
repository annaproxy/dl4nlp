import torch
import torch.nn as nn
from Embedding import Embedder

class Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, bidirectional=False):
        super().__init__()
        self._bidirectional = bidirectional
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        
        self._embedding = Embedder(input_dim, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        self._linear = nn.Linear(hidden_dim, 235)

    def forward(self, inputs):
        
        inputs = self._embedding(inputs)
        
        # use the individual characters for additional classification?
        _, (h_n, _) = self._lstm(inputs)
        h_n = h_n.squeeze(0)
        logits = self._linear(h_n)

        return logits

