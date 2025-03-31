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

def generate_mock_data(seq_length, num_samples, input_size):
    """
    Generates mock data for training.
    """  
    data = np.random.rand(num_samples, seq_length, input_size).astype(np.float32)
    labels = np.random.randint(0, 2, size=(num_samples,))
    return torch.from_numpy(data), torch.from_numpy(labels)

def train_rnn(model, data, labels, num_epochs=10, learning_rate=0.001):
    """
    Trains the RNN model with the given data and labels.
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    input_size = 5
    hidden_size = 10
    output_size = 2
    seq_length = 7
    num_samples = 100
    data, labels = generate_mock_data(seq_length, num_samples, input_size)
    model = SimpleRNN(input_size, hidden_size, output_size)
    train_rnn(model, data, labels)