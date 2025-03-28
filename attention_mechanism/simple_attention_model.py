import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    """
    A simple attention mechanism implementation.
    """
    def __init__(self, input_dim, output_dim):
        super(SimpleAttention, self).__init__()
        self.Wa = nn.Linear(input_dim, output_dim)
        self.Ua = nn.Linear(input_dim, output_dim)
        self.Va = nn.Linear(output_dim, 1)

    def forward(self, query, keys):
        """
        Compute the attention scores and context vector.
        :param query: A tensor of shape (batch_size, input_dim)
        :param keys: A tensor of shape (batch_size, seq_length, input_dim)
        :return: Context vector and attention weights
        """
        query = self.Wa(query)
        keys = self.Ua(keys)
        scores = self.Va(F.tanh(query.unsqueeze(1) + keys))
        attention_weights = F.softmax(scores, dim=1)
        context = torch.sum(attention_weights * keys, dim=1)
        return context, attention_weights

# Example usage
if __name__ == '__main__':
    batch_size = 2
    seq_length = 5
    input_dim = 4
    output_dim = 3
    attention = SimpleAttention(input_dim, output_dim)
    query = torch.rand(batch_size, input_dim)
    keys = torch.rand(batch_size, seq_length, input_dim)
    context, attn_weights = attention(query, keys)
    print('Context vector shape:', context.shape)
    print('Attention weights shape:', attn_weights.shape)