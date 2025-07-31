import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleRNN(nn.Module):
    """Simple RNN model for sequence classification."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class RandomSequenceDataset(Dataset):
    """Generates random sequences for dataset."""
    def __init__(self, num_samples, seq_length, input_size):
        self.data = np.random.rand(num_samples, seq_length, input_size).astype(np.float32)
        self.labels = np.random.randint(0, 2, num_samples)  # Binary classification

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, data_loader, criterion, optimizer, num_epochs):
    """Train the RNN model."""
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    input_size = 10
    hidden_size = 20
    num_classes = 2
    num_samples = 1000
    seq_length = 5
    num_epochs = 5
    batch_size = 16

    dataset = RandomSequenceDataset(num_samples, seq_length, input_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleRNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, data_loader, criterion, optimizer, num_epochs)
    print('Training complete.')