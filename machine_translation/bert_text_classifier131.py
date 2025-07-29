import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            inputs = self.preprocess_data(train_texts)
            labels = torch.tensor(train_labels)
            self.optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, test_texts, test_labels):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_data(test_texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == torch.tensor(test_labels)).float().mean()
            print(f'Accuracy: {accuracy.item() * 100:.2f}%')

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='all')
    train_texts, test_texts, train_labels, test_labels = train_test_split(data.data, data.target, test_size=0.2)
    classifier = TextClassifier()
    classifier.train(train_texts, train_labels)
    classifier.evaluate(test_texts, test_labels)