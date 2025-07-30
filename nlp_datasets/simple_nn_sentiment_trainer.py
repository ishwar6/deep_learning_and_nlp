import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleNN(nn.Module):
    """A simple feedforward neural network for sentiment analysis."""
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

def load_mock_data(batch_size=32):
    """Loads mock data for demonstration purposes."""
    data = torch.randn(1000, 10)
    labels = torch.randint(0, 2, (1000,))
    dataset = torch.utils.data.TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, epochs=5):
    """Trains the model on the provided data loader."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
    dataloader = load_mock_data()
    train_model(model, dataloader)