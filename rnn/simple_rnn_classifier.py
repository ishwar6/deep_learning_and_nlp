import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleRNN(nn.Module):
    """
    A simple RNN model for sequence classification.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class RandomSequenceDataset(Dataset):
    """
    A dataset to generate random sequences and labels for training.
    """
    def __init__(self, num_samples, seq_length, input_size):
        self.data = np.random.rand(num_samples, seq_length, input_size).astype(np.float32)
        self.labels = np.random.randint(0, 2, num_samples).astype(np.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_rnn(model, dataset, num_epochs=5, batch_size=32):
    """
    Trains the RNN model on the provided dataset.
    """
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
    seq_length = 5
    num_samples = 100
    dataset = RandomSequenceDataset(num_samples, seq_length, input_size)
    model = SimpleRNN(input_size, hidden_size, output_size)
    train_rnn(model, dataset)