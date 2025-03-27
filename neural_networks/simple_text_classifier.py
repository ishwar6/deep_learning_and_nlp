import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def load_data():
    data = fetch_20newsgroups(subset='all', categories=['sci.space', 'comp.graphics'])
    return data.data, data.target

def preprocess_data(texts):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(texts).toarray(), vectorizer

def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

def main():
    texts, labels = load_data()
    X, vectorizer = preprocess_data(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = SimpleNN(input_dim=X_train.shape[1], output_dim=len(set(labels)))
    train_model(model, X_train, y_train)
    print('Model training complete.')

if __name__ == '__main__':
    main()