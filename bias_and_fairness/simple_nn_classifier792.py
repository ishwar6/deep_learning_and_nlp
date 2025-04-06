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

def generate_data(samples=1000, features=20):
    """Generates synthetic binary classification data."""
    X, y = make_classification(n_samples=samples, n_features=features, n_classes=2, random_state=42)
    return X, y

def preprocess_data(X, y):
    """Standardizes the feature data and splits it into training and testing sets."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """Trains the neural network model on the training data."""
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

X, y = generate_data()
X_train, X_test, y_train, y_test = preprocess_data(X, y)
train_tensor = torch.FloatTensor(X_train)
target_tensor = torch.FloatTensor(y_train)
train_data = torch.utils.data.TensorDataset(train_tensor, target_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

model = SimpleNN(input_size=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer, epochs=10)

with torch.no_grad():
    model.eval()
    test_tensor = torch.FloatTensor(X_test)
    predictions = model(test_tensor).round()
    accuracy = (predictions.squeeze() == torch.FloatTensor(y_test)).float().mean().item()
    print(f'Accuracy on test set: {accuracy:.4f}')