import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

class TextDataset(Dataset):
    """Custom Dataset for loading text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """A simple feedforward neural network for text classification."""
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def train_model(model, dataloader, criterion, optimizer, epochs):
    """Train the neural network model."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(texts.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

def main():
    newsgroups = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space'])
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(newsgroups.data).toarray()
    y = newsgroups.target
    dataset = TextDataset(torch.tensor(X), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleNN(input_dim=1000, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dataloader, criterion, optimizer, epochs=5)

if __name__ == '__main__':
    main()