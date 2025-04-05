import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return torch.FloatTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(X_test), torch.FloatTensor(y_test)

def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.view(-1), y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def main():
    X_train, y_train, X_test, y_test = load_data()
    model = SimpleNN(input_dim=20, output_dim=1)
    train_model(model, X_train, y_train)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_predictions = torch.sigmoid(test_outputs).view(-1).numpy()
        print('Test Predictions:', test_predictions)

if __name__ == '__main__':
    main()