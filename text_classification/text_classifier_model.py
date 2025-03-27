import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class TextClassifier(nn.Module):
    """Defines a simple feedforward neural network for text classification."""
    def __init__(self, input_size, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

def load_data():
    """Loads and preprocesses the 20 Newsgroups dataset."""
    newsgroups = fetch_20newsgroups(subset='all')
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(newsgroups.data).toarray()
    le = LabelEncoder()
    y = le.fit_transform(newsgroups.target)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    """Trains the model on the training set."""
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    model = TextClassifier(input_size=X_train.shape[1], num_classes=len(set(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    train_loader = [(train_data, train_labels)]
    train_model(model, train_loader, criterion, optimizer)
    print('Training complete.')