import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def create_dataset():
    """
    Creates a mock binary classification dataset.
    """
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    """
    Trains the neural network model.
    """
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

X_train, X_test, y_train, y_test = create_dataset()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
train_dataset = torch.utils.data.TensorDataset(train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

model = SimpleNN(input_size=20, hidden_size=10, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, criterion, optimizer, train_loader)

with torch.no_grad():
    model.eval()
    test_tensor = torch.FloatTensor(X_test)
    predictions = model(test_tensor)
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = (predicted_classes.numpy() == y_test).mean()
    print(f'Test Accuracy: {accuracy:.4f}')