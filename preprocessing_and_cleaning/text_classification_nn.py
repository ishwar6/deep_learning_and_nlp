import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for text classification.
    """
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocess_data():
    """
    Fetches and preprocesses the 20 Newsgroups dataset.
    """
    newsgroups = fetch_20newsgroups(subset='all', categories=None)
    X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    return X_train_vec, X_test_vec, y_train, y_test

def train_model(X_train, y_train):
    """
    Trains the neural network model on the training data.
    """
    input_size = X_train.shape[1]
    num_classes = len(set(y_train))
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
    return model

if __name__ == '__main__':
    X_train_vec, X_test_vec, y_train, y_test = preprocess_data()
    model = train_model(X_train_vec, y_train)
    print('Training complete.')