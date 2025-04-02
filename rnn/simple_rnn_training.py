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
        """
        Forward pass through the RNN and fully connected layer.
        """
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def generate_mock_data(num_samples, seq_length, input_size):
    """
    Generates random mock data for RNN training.
    """
    return torch.rand(num_samples, seq_length, input_size), torch.randint(0, 2, (num_samples,))

def train_rnn_model():
    """
    Trains the RNN model on mock data and prints the loss.
    """
    input_size = 10
    hidden_size = 20
    output_size = 2
    num_samples = 100
    seq_length = 5
    learning_rate = 0.01

    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    x_train, y_train = generate_mock_data(num_samples, seq_length, input_size)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train_rnn_model()