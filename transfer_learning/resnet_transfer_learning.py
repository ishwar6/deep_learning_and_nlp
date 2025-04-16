import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class TransferLearningModel:
    def __init__(self, num_classes):
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def get_model(self):
        return self.model

    def train(self, dataloader, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root='path/to/data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TransferLearningModel(num_classes=len(dataset.classes)).get_model()
    model.train()  # Move model to training mode
    trainer = TransferLearningModel(num_classes=len(dataset.classes))
    trainer.train(dataloader, epochs=5)

if __name__ == '__main__':
    main()