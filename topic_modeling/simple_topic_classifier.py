import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def load_data():
    data = fetch_20newsgroups(subset='all', categories=['sci.space', 'comp.graphics'])
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

def train_model(X_train, y_train, input_dim, output_dim, epochs=100):
    model = SimpleNN(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model

def main():
    X_train, X_test, y_train, y_test, vectorizer = load_data()
    model = train_model(X_train, y_train, X_train.shape[1], len(set(y_train)), epochs=50)
    print('Training complete!')

if __name__ == '__main__':
    main()