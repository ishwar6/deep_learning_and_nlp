import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

def generate_data(samples=1000, features=20, random_state=42):
    X, y = make_classification(n_samples=samples, n_features=features, n_informative=10, random_state=random_state)
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()
    return model

def main():
    X_train, X_test, y_train, y_test = generate_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train_tensor = torch.FloatTensor(X_train)
    labels_tensor = torch.FloatTensor(y_train)
    train_data = torch.utils.data.TensorDataset(train_tensor, labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    model = SimpleNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trained_model = train_model(model, train_loader, criterion, optimizer)
    print('Model trained successfully.')

if __name__ == '__main__':
    main()