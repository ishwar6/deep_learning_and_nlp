import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleRNN(nn.Module):
    """
    A simple RNN model for sequence prediction.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])
        return output

class SequenceDataset(Dataset):
    """
    A simple dataset for sequence data.
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]

def train_model(model, data_loader, criterion, optimizer, num_epochs=10):
    """
    Trains the RNN model on the provided dataset.
    """
    model.train()
    for epoch in range(num_epochs):
        for sequences, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(sequences.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    sequences = torch.randn(100, 10, 5)  # 100 samples, 10 time steps, 5 features
    labels = torch.randn(100, 1)  # 100 labels
    dataset = SequenceDataset(sequences, labels)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleRNN(input_size=5, hidden_size=10, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, data_loader, criterion, optimizer, num_epochs=5)
    print('Training complete.')