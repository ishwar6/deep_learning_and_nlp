import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleNN(nn.Module):
    """A simple feedforward neural network for regression tasks."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_mock_data(num_samples, input_size):
    """Generates mock data for training the neural network."""
    X = np.random.rand(num_samples, input_size).astype(np.float32)
    y = (X.sum(axis=1) + np.random.normal(0, 0.1, num_samples)).astype(np.float32)
    return X, y

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    """Trains the neural network model."""
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    input_size = 10
    hidden_size = 5
    output_size = 1
    num_samples = 100
    num_epochs = 20
    learning_rate = 0.01

    X, y = generate_mock_data(num_samples, input_size)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_model(model, dataloader, criterion, optimizer, num_epochs)
    print('Training complete.')