import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class SampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_rnn_model():
    input_size = 10
    hidden_size = 20
    output_size = 1
    num_samples = 100
    sequence_length = 5

    data = torch.randn(num_samples, sequence_length, input_size)
    labels = torch.randn(num_samples, output_size)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
    train_dataset = SampleDataset(train_data, train_labels)
    test_dataset = SampleDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for inputs, target in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

    print('Training complete')
    return model

if __name__ == '__main__':
    trained_model = train_rnn_model()