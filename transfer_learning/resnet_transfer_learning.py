import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

class TransferLearningModel:
    """
    Class to implement a transfer learning model using a pre-trained ResNet.
    """

    def __init__(self, num_classes):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, dataloader, num_epochs=5):
        """
        Trains the model on the provided dataloader.
        """
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

def main():
    """
    Main function to execute training with mock data.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.FakeData(transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransferLearningModel(num_classes=10)
    model.train(dataloader, num_epochs=5)

if __name__ == '__main__':
    main()