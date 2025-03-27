import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleRNN(nn.Module):
    """
    A simple RNN model for sequence prediction tasks.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])
        return output

class RandomDataset(Dataset):
    """
    A dataset that generates random sequences and labels.
    """
    def __init__(self, num_samples, seq_length, input_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sequence = torch.randn(self.seq_length, self.input_size)
        label = torch.randint(0, 2, (1,)).float()
        return sequence, label

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    """
    Train the RNN model.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences.unsqueeze(0))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

if __name__ == '__main__':
    input_size = 10
    hidden_size = 20
    output_size = 1
    num_samples = 100
    seq_length = 5
    num_epochs = 10

    dataset = RandomDataset(num_samples, seq_length, input_size)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dataloader, criterion, optimizer, num_epochs)
    print('Training complete.')