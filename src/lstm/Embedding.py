import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, input_dim, embedding_dim, train_embedding=True, load_embeddings=True):
        super().__init__()

        self._embedding = nn.Embedding(input_dim, embedding_dim)
        if load_embeddings:
            weight_path = "./models/embeddings/character_embeddings_256.pt"
            print("Pre-Trained Embeddings Loaded")
            self._embedding.load_state_dict({'weight':torch.load(weight_path)})
        self._embedding.weight.requires_grad = train_embedding

    def forward(self, inputs):
        emb = self._embedding(inputs)
        return emb
