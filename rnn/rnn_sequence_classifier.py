import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleDataset(Dataset):
    """Custom dataset for RNN training."""
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class RNNModel(nn.Module):
    """Simple RNN model for sequence classification."""
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """Train the RNN model."""
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def main():
    """Main function to run the training process."""
    np.random.seed(0)
    torch.manual_seed(0)

    data = np.random.rand(100, 10, 5).astype(np.float32)
    targets = np.random.randint(0, 2, size=(100,)).astype(np.long)
    dataset = SimpleDataset(data, targets)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    input_size = 5
    hidden_size = 10
    output_size = 2
    num_epochs = 5

    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs)

if __name__ == '__main__':
    main()