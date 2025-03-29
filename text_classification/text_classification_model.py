import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class TextDataset(Dataset):
    """Custom Dataset for loading text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class TextClassifier(nn.Module):
    """Simple feedforward neural network for text classification."""
    def __init__(self, input_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def train_model(model, dataloader, criterion, optimizer, epochs):
    """Train the text classification model."""
    model.train()
    for epoch in range(epochs):
        for texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print('Training complete.')

if __name__ == '__main__':
    texts = ['This is a positive example.', 'This is a negative example.'] * 50
    labels = [1, 0] * 50
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    y = torch.tensor(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_dataset = TextDataset(torch.tensor(X_train, dtype=torch.float32), y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model = TextClassifier(input_dim=X.shape[1], output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, epochs=5)
    print('Model summary:')
    print(model)