import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

def load_data():
    data = fetch_20newsgroups(subset='all')
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    return X_train_tfidf, X_test_tfidf, y_train, y_test

def train_model(X_train, y_train, input_size, num_classes):
    model = SimpleNN(input_size, num_classes)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(torch.FloatTensor(X_train))
        loss = criterion(output, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return model

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, 1000, 20)
    print('Training completed.')

if __name__ == '__main__':
    main()