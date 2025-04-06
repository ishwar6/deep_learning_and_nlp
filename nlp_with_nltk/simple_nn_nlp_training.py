import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

class SimpleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def preprocess_data(sentences, labels):
    tokenized = [word_tokenize(sentence.lower()) for sentence in sentences]
    return tokenized, labels

def train_model(model, criterion, optimizer, dataloader, epochs):
    for epoch in range(epochs):
        for texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(texts.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    sentences = ['Hello world', 'Deep learning with NLTK', 'Natural language processing is fun']
    labels = [0, 1, 1]
    tokenized_texts, labels = preprocess_data(sentences, labels)
    input_size = 5  # Mock input size
    output_size = 2  # Mock output size
    dataset = SimpleDataset(torch.rand(len(tokenized_texts), input_size), torch.tensor(labels))
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = SimpleNN(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, criterion, optimizer, train_loader, epochs=5)