import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    """Implements a simple scaled dot-product attention mechanism."""
    def __init__(self, embed_size, heads):
        super(SimpleAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N = x.shape[0]
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        energy = torch.einsum("qhd,khd->qhk", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("qhk,vhd->qhd", [attention, values]).reshape(N, query_len, self.embed_size)
        return self.fc_out(out)

# Mock data for testing the attention mechanism
if __name__ == '__main__':
    embed_size = 8
    heads = 2
    attention_layer = SimpleAttention(embed_size, heads)
    mock_input = torch.rand((5, 10, embed_size))
    output = attention_layer(mock_input)
    print(output.shape)  # Expected output shape: (5, 10, 8)