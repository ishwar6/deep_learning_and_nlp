import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

class RNNModel(nn.Module):
    """A simple RNN model for sequence classification."""
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the RNN model."""
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def generate_mock_data(seq_length, num_samples, input_size):
    """Generates random data for mock training and testing."""
    X = np.random.rand(num_samples, seq_length, input_size).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.long)
    return X, y

def train_model(model, data_loader, criterion, optimizer, num_epochs):
    """Trains the RNN model on the provided data loader."""
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print(f'Training complete after {num_epochs} epochs.')

def main():
    """Main function to execute the training of the RNN model."""
    input_size = 10
    hidden_size = 20
    output_size = 2
    seq_length = 5
    num_samples = 100
    num_epochs = 10

    X, y = generate_mock_data(seq_length, num_samples, input_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_tensor = torch.tensor(X_train)
    train_labels = torch.tensor(y_train)
    test_tensor = torch.tensor(X_test)
    test_labels = torch.tensor(y_test)

    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_tensor, train_labels), batch_size=16, shuffle=True)

    train_model(model, train_loader, criterion, optimizer, num_epochs)
    print(model)

if __name__ == '__main__':
    main()