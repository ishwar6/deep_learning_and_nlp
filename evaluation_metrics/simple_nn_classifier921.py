import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for binary classification.
    """
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train_model(model, X_train, y_train, num_epochs=100, learning_rate=0.01):
    """
    Train the neural network model.
    """
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        inputs = torch.FloatTensor(X_train)
        labels = torch.FloatTensor(y_train).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and return accuracy.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_test)
        outputs = model(inputs)
        predicted = (outputs.numpy() > 0.5).astype(int)
    accuracy = accuracy_score(y_test, predicted)
    return accuracy

if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SimpleNN(input_size=X.shape[1])
    trained_model = train_model(model, X_train, y_train)
    accuracy = evaluate_model(trained_model, X_test, y_test)
    print(f'Accuracy of the model: {accuracy:.4f}')