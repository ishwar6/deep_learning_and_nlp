import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention.
    """
    def __init__(self, dropout_rate=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        """
        Computes attention scores and outputs weighted values.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism.
    """
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout_rate)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (num_heads, depth).
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        """
        Executes multi-head attention.
        """
        batch_size = query.size(0)
        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)
        output, attention_weights = self.attention(query, key, value, mask)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        return self.dense(output), attention_weights

# Example usage
if __name__ == '__main__':
    batch_size = 2
    seq_length = 5
    d_model = 64
    num_heads = 8
    query = torch.rand((batch_size, seq_length, d_model))
    key = torch.rand((batch_size, seq_length, d_model))
    value = torch.rand((batch_size, seq_length, d_model))
    attention_layer = MultiHeadAttention(num_heads, d_model)
    output, attn_weights = attention_layer(query, key, value)
    print('Output shape:', output.shape)
    print('Attention weights shape:', attn_weights.shape)