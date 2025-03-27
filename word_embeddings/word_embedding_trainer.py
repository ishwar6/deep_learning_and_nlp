import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input):
        return self.embeddings(input)

def preprocess_data():
    newsgroups = fetch_20newsgroups(subset='all')
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer

def train_model(X_train, y_train, vocab_size, embedding_dim=50, epochs=5):
    model = WordEmbeddingModel(vocab_size, embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(len(X_train.toarray())):
            inputs = torch.tensor(X_train[i].toarray(), dtype=torch.float32).view(-1)
            target = torch.tensor([y_train[i]], dtype=torch.long)
            optimizer.zero_grad()
            output = model(inputs.long())
            loss = loss_function(output.view(1, -1), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')
    return model

if __name__ == '__main__':
    (X_train, X_test, y_train, y_test), vectorizer = preprocess_data()
    vocab_size = len(vectorizer.vocabulary_)
    model = train_model(X_train, y_train, vocab_size)
    print('Training complete. Model ready for inference.')