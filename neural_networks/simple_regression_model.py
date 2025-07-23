import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    """Custom dataset for a simple regression task."""
    def __init__(self, size=100):
        self.x = np.random.rand(size, 1).astype(np.float32)
        self.y = (self.x * 2 + 1 + np.random.normal(0, 0.1, size=(size, 1))).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class SimpleNN(nn.Module):
    """A simple feedforward neural network for regression."""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, data_loader, criterion, optimizer, epochs=10):
    """Trains the model using the provided data loader and optimizer."""
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    dataset = SimpleDataset(size=200)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, data_loader, criterion, optimizer, epochs=20)
    print('Training complete.')