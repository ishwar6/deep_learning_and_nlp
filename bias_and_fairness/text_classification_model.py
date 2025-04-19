import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer, BertForSequenceClassification

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            inputs = self.preprocess_data(texts)
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    def evaluate(self, texts):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_data(texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            return predictions.numpy()

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='all')
    texts, labels = newsgroups.data, newsgroups.target
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.1)
    classifier = TextClassifier()
    classifier.train(texts_train, torch.tensor(labels_train))
    predictions = classifier.evaluate(texts_val)
    print(predictions)