import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

nltk.download('punkt')

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocess_data():
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = data.data
    labels = data.target
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    return tokenized_texts, labels

def create_datasets(tokenized_texts, labels):
    input_size = 10000  
    word_freq = nltk.FreqDist([word for text in tokenized_texts for word in text])
    vocabulary = {word: i for i, (word, _) in enumerate(word_freq.most_common(input_size))}
    encoded_texts = [[vocabulary.get(word, 0) for word in text] for text in tokenized_texts]
    padded_texts = nn.utils.rnn.pad_sequence([torch.tensor(text) for text in encoded_texts], batch_first=True)
    x_train, x_test, y_train, y_test = train_test_split(padded_texts, labels, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train, input_size, hidden_size, output_size, epochs=5):
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train.float())
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return model

if __name__ == '__main__':
    tokenized_texts, labels = preprocess_data()
    x_train, x_test, y_train, y_test = create_datasets(tokenized_texts, labels)
    model = train_model(x_train, y_train, input_size=10000, hidden_size=128, output_size=len(set(labels)))
    print('Training complete!')