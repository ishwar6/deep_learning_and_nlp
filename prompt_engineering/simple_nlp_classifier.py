import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def preprocess_data():
    data = fetch_20newsgroups(subset='all', categories=['sci.space', 'comp.graphics'])
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data.data).toarray()
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data()
    model = SimpleNN(input_dim=X_train.shape[1], output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_loader = [(torch.tensor(X_train[i:i+32]), torch.tensor(y_train[i:i+32])) for i in range(0, len(X_train), 32)]
    train_model(model, train_loader, criterion, optimizer)
    print('Model training complete.')