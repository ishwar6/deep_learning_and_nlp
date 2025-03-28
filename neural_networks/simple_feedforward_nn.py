import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    """Custom dataset for demonstration purposes."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """A simple feedforward neural network with two layers."""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_model(model, dataloader, criterion, optimizer, epochs=5):
    """Train the neural network model."""
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    mock_data = np.random.rand(100, 10).astype(np.float32)
    mock_labels = np.random.rand(100).astype(np.float32)
    dataset = SimpleDataset(mock_data, mock_labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, dataloader, criterion, optimizer, epochs=5)