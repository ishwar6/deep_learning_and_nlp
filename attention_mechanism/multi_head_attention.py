import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
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
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, query_len, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum('qhd,khd->hqk', [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum('hvl,hqk->qvl', [values, attention]).reshape(N, query_len, self.embed_size)
        return self.fc_out(out)

# Example usage
if __name__ == '__main__':
    embed_size = 256
    heads = 8
    batch_size = 10
    seq_length = 20
    x = torch.rand((batch_size, seq_length, embed_size))
    multihead_attention = MultiHeadAttention(embed_size, heads)
    output = multihead_attention(x)
    print(output.shape)