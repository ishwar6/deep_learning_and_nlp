import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    """Custom dataset for simple regression tasks."""
    def __init__(self, num_samples=100):
        self.x = np.random.rand(num_samples, 1).astype(np.float32)
        self.y = (self.x * 2 + 1 + np.random.normal(0, 0.1, (num_samples, 1))).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleLinearRegression(nn.Module):
    """A simple linear regression model."""
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

def train_model(model, data_loader, criterion, optimizer, epochs=100):
    """Train the linear regression model."""
    for epoch in range(epochs):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    dataset = SimpleDataset(num_samples=200)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleLinearRegression()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model, data_loader, criterion, optimizer, epochs=100)
    print('Training complete.')