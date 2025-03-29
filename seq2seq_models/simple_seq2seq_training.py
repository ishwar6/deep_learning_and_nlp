import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

class SimpleSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SimpleSeq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, output_dim, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        output, _ = self.decoder(hidden.unsqueeze(1))
        return output

class RandomDataset(Dataset):
    def __init__(self, length, input_dim, output_dim, seq_length):
        self.length = length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.rand(self.seq_length, self.input_dim)
        y = torch.rand(self.seq_length, self.output_dim)
        return x, y

def train_model(model, data_loader, num_epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

if __name__ == '__main__':
    input_dim = 10
    output_dim = 10
    hidden_dim = 20
    seq_length = 5
    dataset = RandomDataset(100, input_dim, output_dim, seq_length)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SimpleSeq2Seq(input_dim, output_dim, hidden_dim)
    train_model(model, data_loader)