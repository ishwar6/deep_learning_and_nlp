import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class SimpleNN(nn.Module):
    """A simple feedforward neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RandomDataset(Dataset):
    """A dataset that generates random data for training and testing."""
    def __init__(self, num_samples, input_size):
        self.data = torch.rand(num_samples, input_size)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """Trains the given model using the provided data loader."""
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    input_size = 20
    hidden_size = 10
    output_size = 1
    num_samples = 1000
    batch_size = 32
    num_epochs = 5

    dataset = RandomDataset(num_samples, input_size)
    train_data, _ = train_test_split(dataset, test_size=0.2)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs)
    print('Training complete.')