import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class RandomSequenceDataset(Dataset):
    """
    Generates a dataset of random sequences and labels.
    """
    def __init__(self, num_samples, seq_length, input_size):
        self.data = np.random.rand(num_samples, seq_length, input_size).astype(np.float32)
        self.labels = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_rnn_model():
    """
    Trains the RNN model on generated random data.
    """
    input_size = 10
    hidden_size = 20
    output_size = 1
    num_samples = 1000
    seq_length = 5
    batch_size = 32

    dataset = RandomSequenceDataset(num_samples, seq_length, input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train_rnn_model()