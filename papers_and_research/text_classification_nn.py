import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class TextDataset(Dataset):
    """
    A PyTorch Dataset for loading text data.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for text classification.
    """
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, dataloader, criterion, optimizer, epochs):
    """
    Trains the model using the provided DataLoader.
    """
    model.train()
    for epoch in range(epochs):
        for texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='all')
    X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(max_features=1000)
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
    X_test_vectorized = vectorizer.transform(X_test).toarray()
    train_dataset = TextDataset(torch.FloatTensor(X_train_vectorized), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = SimpleNN(input_size=1000, output_size=len(newsgroups.target_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    print('Training complete.')