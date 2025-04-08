import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    """
    A simple dataset class that generates random data.
    """
    def __init__(self, size):
        self.size = size
        self.data = np.random.rand(size, 10).astype(np.float32)
        self.labels = (self.data.sum(axis=1) > 5).astype(np.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network model.
    """
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def train_model(model, data_loader, criterion, optimizer, epochs):
    """
    Train the neural network model.
    """
    model.train()
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

def main():
    """
    Main function to set up the dataset, model, and initiate training.
    """
    dataset = SimpleDataset(size=1000)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, data_loader, criterion, optimizer, epochs=10)

if __name__ == '__main__':
    main()