import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    """
    A simple dataset for RNN training.
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class RNNModel(nn.Module):
    """
    A simple RNN model with an embedding layer and a linear output layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    """
    Train the RNN model.
    """
    model.train()
    for epoch in range(num_epochs):
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    input_size = 10
    hidden_size = 20
    output_size = 2
    num_samples = 100
    sequence_length = 5
    sequences = torch.randint(0, input_size, (num_samples, sequence_length))
    labels = torch.randint(0, output_size, (num_samples,))
    dataset = SimpleDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dataloader, criterion, optimizer, num_epochs=5)
    print('Training complete.')