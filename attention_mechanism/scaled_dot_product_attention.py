import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Implements a simple scaled dot-product attention mechanism."""
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.values = nn.Linear(d_model, d_model, bias=False)
        self.keys = nn.Linear(d_model, d_model, bias=False)
        self.queries = nn.Linear(d_model, d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        N, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)
        values = values.view(N, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        energy = torch.einsum('nhqd,nhkd->nhqk', queries, keys)
        attention = F.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        out = torch.einsum('nhql,nhld->nhqd', attention, values).reshape(N, seq_length, self.d_model)
        return self.fc_out(out)

# Example usage
if __name__ == '__main__':
    d_model = 128
    n_heads = 8
    seq_length = 10
    batch_size = 2
    x = torch.rand(batch_size, seq_length, d_model)
    attention_layer = Attention(d_model, n_heads)
    output = attention_layer(x)
    print('Output shape:', output.shape)