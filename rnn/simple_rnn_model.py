import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleRNN(nn.Module):
    """A simple RNN model for sequence prediction."""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the RNN and fully connected layer."""
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_rnn(model, data, targets, epochs=100, learning_rate=0.01):
    """Train the RNN model using mean squared error loss."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    input_size = 1
    hidden_size = 5
    output_size = 1
    seq_length = 10
    data = torch.Tensor(np.random.rand(100, seq_length, input_size))
    targets = torch.Tensor(np.random.rand(100, output_size))
    model = SimpleRNN(input_size, hidden_size, output_size)
    train_rnn(model, data, targets)