import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

class RNNTextClassifier(nn.Module):
    """Simple RNN model for text classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

class TextDataset(Dataset):
    """Custom dataset for loading text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def preprocess_data():
    """Fetches and preprocesses the 20 Newsgroups dataset."""
    newsgroups = fetch_20newsgroups(subset='all')
    texts = newsgroups.data
    labels = newsgroups.target
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return texts, labels, len(le.classes_)

def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    texts, labels = zip(*batch)
    lengths = [len(text.split()) for text in texts]
    padded_texts = [text.split() + ['<PAD>'] * (max(lengths) - len(text.split())) for text in texts]
    return torch.tensor(padded_texts, dtype=torch.long), torch.tensor(labels)

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """Train the RNN model with the provided data loader."""
    model.train()
    for epoch in range(num_epochs):
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    texts, labels, output_dim = preprocess_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_dataset = TextDataset(train_texts, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model = RNNTextClassifier(vocab_size=10000, embedding_dim=100, hidden_dim=50, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer)