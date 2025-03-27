import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.word2idx = self.build_vocab(texts)

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        return {word: idx for idx, (word, _) in enumerate(counter.items())}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx].split()
        indices = [self.word2idx[word] for word in text]
        label = self.labels[idx]
        return torch.tensor(indices), torch.tensor(label)

class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(WordEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, input):
        return self.embeddings(input)

def train_model(model, data_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}')

if __name__ == '__main__':
    newsgroup_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts, labels = newsgroup_data.data, newsgroup_data.target
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2)
    train_dataset = TextDataset(texts_train, labels_train)
    val_dataset = TextDataset(texts_val, labels_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = WordEmbeddingModel(vocab_size=len(train_dataset.word2idx), embed_size=100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    print('Training complete')