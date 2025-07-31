import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class ZeroShotClassifier:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=20)

    def preprocess(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        inputs = self.preprocess(texts)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        inputs = self.preprocess(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='all')
    texts = newsgroups.data[:100]
    labels = newsgroups.target[:100]
    classifier = ZeroShotClassifier()
    classifier.train(texts, torch.tensor(labels))
    test_texts = ['I love programming with Python!', 'The stock market is volatile.']
    predictions = classifier.predict(test_texts)
    print('Predictions:', predictions)