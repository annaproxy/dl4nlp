import torch
import torch.nn as nn
from Embedding import Embedder

class Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, bidirectional=False):
        super().__init__()
        self._bidirectional = 2 if bidirectional else 1
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._linear_dim = self._hidden_dim*self._num_layers*self._bidirectional
        
        self._embedding = Embedder(input_dim, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, 
                             hidden_dim, 
                             num_layers, 
                             bidirectional=bidirectional, 
                             batch_first=True, dropout=0.4)
        
        self._linear = nn.Linear(self._linear_dim, 235)

    def forward(self, inputs):
        
        inputs = self._embedding(inputs)
        
        # use the individual characters for additional classification?
        _, (h_n, _) = self._lstm(inputs)
        h_n = h_n.permute(1,0,2).contiguous().view(h_n.shape[1], self._linear_dim)
        logits = self._linear(h_n)

        return logits

