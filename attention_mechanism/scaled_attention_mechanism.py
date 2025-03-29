import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Implements the scaled dot-product attention mechanism.
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            k: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            v: Value tensor of shape (batch_size, num_heads, seq_len, d_v)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len, seq_len)
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

# Simulating a simple scenario
if __name__ == '__main__':
    batch_size = 2
    num_heads = 1
    seq_len = 3
    d_k = d_v = 4
    q = torch.rand(batch_size, num_heads, seq_len, d_k)
    k = torch.rand(batch_size, num_heads, seq_len, d_k)
    v = torch.rand(batch_size, num_heads, seq_len, d_v)
    mask = torch.ones(batch_size, 1, seq_len, seq_len).bool()
    attention = ScaledDotProductAttention()
    output, attn_weights = attention(q, k, v, mask)
    print("Output:", output)
    print("Attention Weights:", attn_weights)