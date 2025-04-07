import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleDataset(Dataset):
    """Custom dataset for loading features and labels."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """A simple feedforward neural network."""
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, dataloader, epochs=5):
    """Train the neural network model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    np.random.seed(0)
    features = np.random.rand(1000, 10)
    labels = np.random.rand(1000, 1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    train_dataset = SimpleDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = SimpleNN(input_size=10)
    train_model(model, train_loader, epochs=10)
    print("Training complete.")