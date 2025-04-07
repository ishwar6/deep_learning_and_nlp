import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

class TransferLearningModel:
    def __init__(self, num_classes):
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def train(self, train_loader, criterion, optimizer, num_epochs=5):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total:.2f}%')

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.FakeData(transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = datasets.FakeData(transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = TransferLearningModel(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.fc.parameters(), lr=0.001)
    model.train(train_loader, criterion, optimizer, num_epochs=5)
    model.evaluate(test_loader)

if __name__ == '__main__':
    main()