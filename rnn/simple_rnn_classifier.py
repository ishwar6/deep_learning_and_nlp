import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleRNN(nn.Module):
    """
    A simple RNN model for sequence classification.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])
        return out

class SampleDataset(Dataset):
    """
    A simple dataset class for generating mock data.
    """
    def __init__(self, num_samples, seq_length, input_size, output_size):
        self.data = torch.rand(num_samples, seq_length, input_size)
        self.labels = torch.randint(0, output_size, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    """
    Train the RNN model on the dataset.
    """
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
    output_size = 5
    num_samples = 100
    seq_length = 15
    num_epochs = 5

    dataset = SampleDataset(num_samples, seq_length, input_size, output_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs)