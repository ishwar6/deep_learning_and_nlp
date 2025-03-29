import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleRNN(nn.Module):
    """A simple RNN model for sequence classification."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass for the RNN."""
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class RandomSequenceDataset(Dataset):
    """A dataset that generates random sequences for training."""
    def __init__(self, num_samples, seq_length, input_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.input_size = input_size
        self.data = torch.randn(num_samples, seq_length, input_size)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_rnn(model, dataloader, criterion, optimizer, num_epochs=5):
    """Train the RNN model using the provided data loader."""
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
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
    batch_size = 5
    num_samples = 100
    seq_length = 15
    num_epochs = 3
    dataset = RandomSequenceDataset(num_samples, seq_length, input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_rnn(model, dataloader, criterion, optimizer, num_epochs)