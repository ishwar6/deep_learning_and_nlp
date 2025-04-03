import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

def preprocess_data():
    data = fetch_20newsgroups(subset='all')
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data.data).toarray()
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def main():
    X_train, X_test, y_train, y_test = preprocess_data()
    input_size = X_train.shape[1]
    num_classes = len(set(y_train))
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = torch.utils.data.DataLoader(list(zip(torch.FloatTensor(X_train), torch.LongTensor(y_train))), batch_size=32)
    train_model(model, train_loader, criterion, optimizer)
    print('Training complete.')

if __name__ == '__main__':
    main()