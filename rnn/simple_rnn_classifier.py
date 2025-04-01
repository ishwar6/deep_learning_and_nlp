import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNNModel(nn.Module):
    """
    A simple RNN model for sequence classification.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])
        return output

def generate_mock_data(seq_length, num_samples, input_size):
    """
    Generates mock data for RNN training.
    """
    X = np.random.rand(num_samples, seq_length, input_size).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.long)
    return torch.from_numpy(X), torch.from_numpy(y)

def train_rnn_model(model, data_loader, criterion, optimizer, num_epochs):
    """
    Trains the RNN model with the provided data loader.
    """
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    input_size = 10
    hidden_size = 20
    output_size = 2
    seq_length = 5
    num_samples = 100
    num_epochs = 10

    X, y = generate_mock_data(seq_length, num_samples, input_size)
    data_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=16, shuffle=True)
    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_rnn_model(model, data_loader, criterion, optimizer, num_epochs)
    print('Training complete.')