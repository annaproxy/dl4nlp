import torch
import torch.nn as nn
#from Embedding import Embedder

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


class Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, bidirectional=False, lang_amount = 235, load_embeddings=True):
        super().__init__()
        self._bidirectional = 2 if bidirectional else 1
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._linear_dim = self._hidden_dim*self._num_layers*self._bidirectional

        self._embedding = Embedder(input_dim, embedding_dim, train_embedding=True, load_embeddings=load_embeddings)

        self._lstm = nn.LSTM(embedding_dim,
                             hidden_dim,
                             num_layers,
                             bidirectional=bidirectional,
                             batch_first=True, dropout=0.4)

        self._linear = nn.Linear(self._linear_dim, lang_amount )

    def forward(self, inputs):

        inputs = self._embedding(inputs)

        # use the individual characters for additional classification?
        _, (h_n, _) = self._lstm(inputs)
        h_n = h_n.permute(1,0,2).contiguous().view(h_n.shape[1], self._linear_dim)
        logits = self._linear(h_n)

        return logits
