import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            inputs = self.preprocess_data(train_texts)
            outputs = self.model(**inputs, labels=train_labels)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

    def evaluate(self, test_texts, test_labels):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_data(test_texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == test_labels).float().mean().item()
            print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='all')
    texts, labels = data.data, data.target
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    classifier = TextClassifier(num_labels=len(data.target_names))
    classifier.train(train_texts, train_labels)
    classifier.evaluate(test_texts, test_labels)