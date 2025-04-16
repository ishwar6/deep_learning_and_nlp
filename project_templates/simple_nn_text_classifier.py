import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleDataset(Dataset):
    """Custom dataset for text classification using mock data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """Simple feedforward neural network for classification."""
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, criterion, optimizer, dataloader, num_epochs=5):
    """Train the model and print training loss."""
    for epoch in range(num_epochs):
        total_loss = 0
        for texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(texts.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

# Mock data generation
num_samples = 1000
num_features = 20
num_classes = 2
X = np.random.rand(num_samples, num_features).astype(np.float32)
Y = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
train_dataset = SimpleDataset(torch.tensor(X_train), torch.tensor(Y_train))
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model initialization and training
model = SimpleNN(input_size=num_features, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, criterion, optimizer, dataloader, num_epochs=5)