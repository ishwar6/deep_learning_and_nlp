import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_classes=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)

    def preprocess_data(self, texts, labels):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        return tokens, torch.tensor(labels)

    def train(self, texts, labels, epochs=3):
        self.model.train()
        tokens, labels = self.preprocess_data(texts, labels)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(tokens['input_ids'], attention_mask=tokens['attention_mask'], labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
            _, predicted = torch.max(outputs.logits, dim=1)
        return predicted.numpy()

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='train')
    texts, labels = newsgroups.data, newsgroups.target
    classifier = TextClassifier()
    classifier.train(texts, labels)
    sample_texts = texts[:5]
    predictions = classifier.predict(sample_texts)
    print('Predictions for sample texts:', predictions)