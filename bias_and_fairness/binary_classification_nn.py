import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SimpleNN(nn.Module):
    """A simple feedforward neural network for binary classification."""
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def generate_data(samples=1000, features=20, random_state=42):
    """Generates a synthetic binary classification dataset."""
    X, y = make_classification(n_samples=samples, n_features=features, n_classes=2, random_state=random_state)
    return X, y

def train_model(model, X_train, y_train, epochs=100, lr=0.01):
    """Trains the model on the training data."""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1).float())
        loss.backward()
        optimizer.step()
    return model

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

model = SimpleNN(input_size=X_train.shape[1])
trained_model = train_model(model, X_train_tensor, y_train_tensor)

with torch.no_grad():
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    predictions = trained_model(test_tensor)
    predicted_classes = (predictions.numpy() > 0.5).astype(int)

print(predicted_classes)  
print(predictions.numpy())