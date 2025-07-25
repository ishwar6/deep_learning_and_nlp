import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for binary classification.
    """
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def generate_data(samples=1000, features=20):
    """
    Generates synthetic data for binary classification.
    """
    X, y = make_classification(n_samples=samples, n_features=features, n_informative=10, n_redundant=5, random_state=42)
    return X, y

def train_model(model, X_train, y_train, epochs=100, lr=0.01):
    """
    Trains the neural network model with given data.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train.float())
        loss.backward()
        optimizer.step()  
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on test data.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).round()  
    accuracy = (predictions.squeeze() == y_test).float().mean().item()
    return accuracy

if __name__ == '__main__':
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = SimpleNN(input_size=X_train.shape[1])
    model = train_model(model, torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    accuracy = evaluate_model(model, torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    print(f'Model Accuracy: {accuracy * 100:.2f}%')