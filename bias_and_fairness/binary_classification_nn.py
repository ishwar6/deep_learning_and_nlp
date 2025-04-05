import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
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

def create_dataset(samples=1000, features=20, random_state=42):
    """
    Creates a synthetic binary classification dataset.
    """
    X, y = make_classification(n_samples=samples, n_features=features, n_classes=2,
                               random_state=random_state)
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """
    Trains the model using the provided data loader.
    """
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.view(-1, 1).float())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = create_dataset()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_tensor = torch.FloatTensor(X_train)
    target_tensor = torch.LongTensor(y_train)
    train_data = torch.utils.data.TensorDataset(train_tensor, target_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    model = SimpleNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    print('Model training complete.')