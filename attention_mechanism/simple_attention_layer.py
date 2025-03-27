import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    """
    Implements a simple scaled dot-product attention mechanism.
    """
    def __init__(self, d_model):
        super(SimpleAttention, self).__init__()
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        """
        Compute the attention scores and apply them to values.
        
        Args:
            queries: Tensor of shape (batch_size, num_queries, d_model)
            keys: Tensor of shape (batch_size, num_keys, d_model)
            values: Tensor of shape (batch_size, num_keys, d_model)
        
        Returns:
            Attention output tensor of shape (batch_size, num_queries, d_model)
        """
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, values)
        return output

# Mock data for testing the attention mechanism
batch_size = 2
num_queries = 3
num_keys = 4
d_model = 5
queries = torch.rand(batch_size, num_queries, d_model)
keys = torch.rand(batch_size, num_keys, d_model)
values = torch.rand(batch_size, num_keys, d_model)

# Initialize and run the attention layer
attention_layer = SimpleAttention(d_model)
output = attention_layer(queries, keys, values)
print("Attention output:\n", output)