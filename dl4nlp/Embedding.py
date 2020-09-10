import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, input_dim, embedding_dim, train_embedding=True):
        super().__init__()
        
        self._embedding = nn.Embedding(input_dim, embedding_dim)
        self._embedding.weight.requires_grad = train_embedding
        
    def forward(self, inputs):
        emb = self._embedding(inputs)
        return emb