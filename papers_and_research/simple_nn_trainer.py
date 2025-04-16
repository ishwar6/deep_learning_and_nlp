import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class SimpleDataset(Dataset):
    """Custom dataset for loading sample data."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """A simple feedforward neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, criterion, optimizer, dataloader, epochs=5):
    """Train the neural network model."""
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    data = torch.randn(100, 10)
    labels = (data.sum(dim=1) > 0).long()  # Simple binary classification
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    train_dataset = SimpleDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, criterion, optimizer, train_loader, epochs=10)
    print('Training complete.')