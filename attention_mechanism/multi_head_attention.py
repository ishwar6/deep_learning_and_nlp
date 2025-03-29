import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism.
    """
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
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
        length = x.shape[1]
        embed_size = x.shape[2]
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        queries = queries.view(N, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(N, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.einsum("qhd,khd->qhk", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)
        out = torch.einsum("qhk,khd->qhd", [attention, values]).reshape(N, length, self.embed_size)
        return self.fc_out(out)

# Example usage
if __name__ == '__main__':
    embed_size = 256
    heads = 8
    batch_size = 10
    seq_length = 20
    attention = MultiHeadAttention(embed_size, heads)
    x = torch.rand((batch_size, seq_length, embed_size))
    output = attention(x)
    print(output.shape)