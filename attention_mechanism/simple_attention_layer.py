import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    """Implements a simple attention mechanism based on dot-product attention."""
    def __init__(self):
        super(SimpleAttention, self).__init__()

    def forward(self, query, key, value):
        """Calculates attention scores and outputs weighted values."""
        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

# Simulating data
batch_size = 2
seq_length = 3
embedding_dim = 4
query = torch.rand(batch_size, seq_length, embedding_dim)
key = torch.rand(batch_size, seq_length, embedding_dim)
value = torch.rand(batch_size, seq_length, embedding_dim)

# Using the attention mechanism
attention_layer = SimpleAttention()
output, attention_weights = attention_layer(query, key, value)
print("Output:", output)
print("Attention Weights:", attention_weights)