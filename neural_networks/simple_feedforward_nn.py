import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.
        """ 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class RandomDataset(Dataset):
    """
    A dataset that returns random data points and labels.
    """ 
    def __init__(self, num_samples, input_size):
        self.data = np.random.rand(num_samples, input_size).astype(np.float32)
        self.labels = np.random.randint(0, 2, size=(num_samples,)).astype(np.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    """
    Train the neural network model.
    """ 
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

if __name__ == '__main__':
    input_size = 10
    hidden_size = 5
    output_size = 2
    num_samples = 100
    num_epochs = 10
    batch_size = 16

    dataset = RandomDataset(num_samples, input_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    print('Training complete.')