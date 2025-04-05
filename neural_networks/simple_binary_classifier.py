import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class SimpleDataset(Dataset):
    """Custom dataset for binary classification."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """A simple feedforward neural network for binary classification."""
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

def train_model(model, dataloader, criterion, optimizer, epochs):
    """Train the neural network model."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

if __name__ == '__main__':
    features = torch.randn(1000, 20)
    labels = (torch.randn(1000) > 0).long()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    train_dataset = SimpleDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = SimpleNN(input_size=20)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    print('Training complete.')