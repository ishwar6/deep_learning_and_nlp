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

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3, learning_rate=5e-5):
        train_encodings = self.preprocess_data(train_texts)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(**train_encodings, labels=train_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, test_texts, test_labels):
        test_encodings = self.preprocess_data(test_texts)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**test_encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == test_labels).float().mean().item()
            print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='all')
    texts = newsgroups.data
    labels = newsgroups.target
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    classifier = TextClassifier()
    classifier.train(train_texts, train_labels)
    classifier.evaluate(test_texts, test_labels)