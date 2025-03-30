import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class SimpleDataset(Dataset):
    """Custom Dataset for loading text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """A simple feedforward neural network for text classification."""
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=5):
    """Train the neural network model."""
    model.train()
    for epoch in range(epochs):
        for texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Mock data
texts = torch.rand(100, 20)  # 100 samples, 20 features each
labels = torch.randint(0, 2, (100,))  # Binary labels

# Prepare dataset and dataloader
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
dataset = SimpleDataset(train_texts, train_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=20, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, dataloader, criterion, optimizer, epochs=5)