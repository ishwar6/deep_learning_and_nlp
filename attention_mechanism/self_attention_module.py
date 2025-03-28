import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Implements scaled dot-product self-attention mechanism."""
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, _ = x.shape
        value_len, key_len, query_len = seq_length, seq_length, seq_length
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)

        energy = torch.einsum('qhd,khd->qhk', [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum('qhk,khd->qhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)


# Example usage
if __name__ == '__main__':
    embed_size = 256
    heads = 8
    attention = SelfAttention(embed_size, heads)
    mock_data = torch.rand(2, 10, embed_size)
    output = attention(mock_data)
    print(output.shape)