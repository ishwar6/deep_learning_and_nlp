import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class SimpleTextClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleTextClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def preprocess_data():
    data = fetch_20newsgroups(subset='all')
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return X_train_vec, X_test_vec, y_train_encoded, y_test_encoded, len(label_encoder.classes_)

def train_model(X_train, y_train, input_dim, output_dim):
    model = SimpleTextClassifier(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
    return model

def main():
    X_train, X_test, y_train, y_test, output_dim = preprocess_data()
    input_dim = X_train.shape[1]
    model = train_model(X_train, y_train, input_dim, output_dim)
    print('Model training completed.')

if __name__ == '__main__':
    main()