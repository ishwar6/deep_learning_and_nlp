import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)
        attention_scores = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5), dim=-1)
        attention_output = torch.matmul(attention_scores, value)
        return attention_output

if __name__ == '__main__':
    input_dim = 16
    output_dim = 8
    batch_size = 4
    sequence_length = 10
    mock_data = torch.rand(batch_size, sequence_length, input_dim)
    attention_model = SimpleAttention(input_dim, output_dim)
    output = attention_model(mock_data)
    print("Attention output shape:", output.shape)