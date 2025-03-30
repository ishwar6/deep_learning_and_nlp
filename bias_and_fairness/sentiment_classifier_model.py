import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class SentimentClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def preprocess_data(self, texts, labels):
        encoded_data = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        return encoded_data, torch.tensor(labels)

    def train(self, texts, labels, epochs=3, batch_size=8):
        encoded_data, labels_tensor = self.preprocess_data(texts, labels)
        train_data, val_data, train_labels, val_labels = train_test_split(encoded_data['input_ids'], labels_tensor, test_size=0.2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.model.train()

        for epoch in range(epochs):
            for i in range(0, len(train_data), batch_size):
                optimizer.zero_grad()
                outputs = self.model(train_data[i:i+batch_size], labels=train_labels[i:i+batch_size])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, texts, labels):
        self.model.eval()
        encoded_data, labels_tensor = self.preprocess_data(texts, labels)
        with torch.no_grad():
            outputs = self.model(encoded_data['input_ids'])
            predictions = torch.argmax(outputs.logits, dim=1)
        print(classification_report(labels_tensor.numpy(), predictions.numpy()))

# Sample data
texts = ['I love this product!', 'This is the worst experience.', 'Absolutely fantastic!']
labels = [1, 0, 1]
classifier = SentimentClassifier()
classifier.train(texts, labels)
classifier.evaluate(texts, labels)