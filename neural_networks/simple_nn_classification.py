import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    Generate a synthetic binary classification dataset.
    """
    X, y = make_classification(n_samples=samples, n_features=features, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, criterion, optimizer, train_loader, epochs=10):
    """
    Train the neural network model.
    """
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

X_train, X_test, y_train, y_test = generate_data()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
train_dataset = torch.utils.data.TensorDataset(train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

model = SimpleNN(input_size=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, criterion, optimizer, train_loader)

test_tensor = torch.FloatTensor(X_test)
model.eval()
with torch.no_grad():
    predictions = model(test_tensor).round()
    accuracy = (predictions.squeeze() == torch.FloatTensor(y_test)).float().mean()
    print(f'Accuracy on test set: {accuracy.item()*100:.2f}%')