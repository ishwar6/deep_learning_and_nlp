import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for binary classification.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        return x

def create_mock_data(num_samples=1000):
    """
    Generates mock data for training and testing the CNN.
    """
    X, y = make_classification(n_samples=num_samples, n_features=784,
                               n_classes=2, n_informative=10)
    X = X.reshape(num_samples, 1, 28, 28)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    """
    Trains the SimpleCNN model on mock data and prints the accuracy.
    """
    X_train, X_test, y_train, y_test = create_mock_data()
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        outputs = model(torch.FloatTensor(X_test))
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size
        correct += (predicted.numpy() == y_test).sum()
    print(f'Accuracy: {100 * correct / total}%')

if __name__ == '__main__':
    train_model()