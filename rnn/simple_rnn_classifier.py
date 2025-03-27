import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleRNN(nn.Module):
    """A simple RNN model for sequence classification."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the RNN model."""
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class RandomDataset(Dataset):
    """Generates random sequences for training."""
    def __init__(self, num_samples, seq_length, input_size):
        self.data = torch.randn(num_samples, seq_length, input_size)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, dataset, num_epochs=5, batch_size=32):
    """Trains the RNN model on the provided dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    input_size = 10
    hidden_size = 20
    output_size = 2
    num_samples = 1000
    seq_length = 5
    dataset = RandomDataset(num_samples, seq_length, input_size)
    model = SimpleRNN(input_size, hidden_size, output_size)
    train_model(model, dataset, num_epochs=10, batch_size=16)