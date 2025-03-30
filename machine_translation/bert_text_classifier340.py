import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            inputs = self.preprocess_data(train_texts)
            labels = torch.tensor(train_labels)
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        inputs = self.preprocess_data(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).numpy()

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='train')
    texts, labels = newsgroups.data, newsgroups.target
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
    classifier = TextClassifier()
    classifier.train(train_texts, train_labels)
    predictions = classifier.predict(val_texts)
    print('Predictions:', predictions)