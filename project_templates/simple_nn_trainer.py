import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, device, train_loader, optimizer, criterion, epochs=5):
    """
    Trains the model using the provided data loader.
    """
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

def main():
    """
    Main function to set up data, model, and training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_size = 128
    output_size = 10  # 10 classes for digits 0-9
    batch_size = 64

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_model(model, device, train_loader, optimizer, criterion)

if __name__ == '__main__':
    main()