import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleRNN(nn.Module):
    """
    A simple RNN model for sequence classification.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the RNN.
        """
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def generate_mock_data(num_samples, seq_length, input_size):
    """
    Generate mock data for training.
    """
    X = np.random.rand(num_samples, seq_length, input_size).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.long)
    return torch.tensor(X), torch.tensor(y)

def train_rnn_model(model, data, labels, num_epochs=100, learning_rate=0.01):
    """
    Train the RNN model on the provided data.
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
hidden_size = 5
output_size = 2
num_samples = 100
seq_length = 7
X, y = generate_mock_data(num_samples, seq_length, input_size)
model = SimpleRNN(input_size, hidden_size, output_size)
train_rnn_model(model, X, y)