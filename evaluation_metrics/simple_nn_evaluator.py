import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(model, criterion, optimizer, data_loader, num_epochs=100):
    """
    Train the neural network model.
    """
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, data_loader):
    """
    Evaluate the model on the test set and print the accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_tensor = torch.FloatTensor(X_train)
train_labels = torch.LongTensor(y_train)
train_data = torch.utils.data.TensorDataset(train_tensor, train_labels)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_tensor = torch.FloatTensor(X_test)
test_labels = torch.LongTensor(y_test)
test_data = torch.utils.data.TensorDataset(test_tensor, test_labels)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

model = SimpleNN(input_size=20, hidden_size=10, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, criterion, optimizer, train_loader)
evaluate_model(model, test_loader)