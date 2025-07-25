import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNNModel(nn.Module):
    """
    A simple RNN model for sequence classification.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return out

def generate_mock_data(num_samples, seq_length, input_size):
    """
    Generates random mock data for testing the RNN model.
    """  
    return torch.randn(num_samples, seq_length, input_size), torch.randint(0, 2, (num_samples,))

def train_model(model, data, labels, num_epochs=100, learning_rate=0.001):
    """
    Trains the RNN model using the provided data and labels.
    """  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

input_size = 10
hidden_size = 20
output_size = 2
num_samples = 100
seq_length = 5
mock_data, mock_labels = generate_mock_data(num_samples, seq_length, input_size)
model = RNNModel(input_size, hidden_size, output_size)
train_model(model, mock_data, mock_labels)