import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleDataset(Dataset):
    """Custom dataset for simple numerical data."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleNN(nn.Module):
    """A simple feedforward neural network."""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=5):
    """Train the neural network model."""
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    np.random.seed(0)
    x_data = np.random.rand(100, 10).astype(np.float32)
    y_data = (x_data.sum(axis=1) > 5).astype(np.float32)
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
    train_dataset = SimpleDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = SimpleNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    print('Training complete.')