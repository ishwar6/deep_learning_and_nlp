import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for text classification.
    """
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def load_data():
    """
    Loads and preprocesses the 20 Newsgroups dataset.
    """
    newsgroups = fetch_20newsgroups(subset='all')
    X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    return torch.FloatTensor(X_train_tfidf), torch.LongTensor(y_train), torch.FloatTensor(X_test_tfidf), torch.LongTensor(y_test)

def train_model(model, X_train, y_train, epochs=5, lr=0.01):
    """
    Trains the model on the training data.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

def main():
    """
    Main function to execute the training of the model.
    """
    X_train, y_train, X_test, y_test = load_data()
    model = SimpleNN(input_size=X_train.shape[1], output_size=len(set(y_train.numpy())))
    train_model(model, X_train, y_train)
    print('Training completed.')

if __name__ == '__main__':
    main()