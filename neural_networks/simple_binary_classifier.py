import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleNN(nn.Module):
    """ A simple feedforward neural network for binary classification. """
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class RandomDataset(Dataset):
    """ Generates a random dataset for binary classification. """
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.labels = (torch.rand(size) > 0.5).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, dataset, epochs=10, batch_size=32):
    """ Trains the model on the given dataset. """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}') 

if __name__ == '__main__':
    dataset = RandomDataset(size=1000)
    model = SimpleNN()
    train_model(model, dataset, epochs=10, batch_size=32)