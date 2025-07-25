import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleDataset(Dataset):
    """Custom dataset for text classification."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """Simple feedforward neural network for text classification."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=5):
    """Train the neural network model."""
    model.train()
    for epoch in range(epochs):
        for texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(texts.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    np.random.seed(0)
    texts = np.random.rand(100, 10)
    labels = np.random.randint(0, 2, size=(100,))
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    dataset = SimpleDataset(torch.tensor(X_train), torch.tensor(y_train))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dataloader, criterion, optimizer, epochs=5)