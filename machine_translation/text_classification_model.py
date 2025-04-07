import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class TextClassifier:
    def __init__(self, model_name, num_classes):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return encodings, torch.tensor(labels)

    def train(self, texts, labels, epochs=3):
        self.model.train()
        encodings, labels = self.preprocess_data(texts, labels)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(**encodings, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, texts, labels):
        self.model.eval()
        encodings, labels = self.preprocess_data(texts, labels)
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        print(classification_report(labels.numpy(), predictions.numpy()))

# Mock data
texts = ['I love machine learning!', 'Deep learning is fascinating.', 'Natural language processing is powerful.']
labels = [1, 1, 1]

# Initialize and train the model
classifier = TextClassifier('bert-base-uncased', num_classes=2)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
classifier.train(train_texts, train_labels)
classifier.evaluate(val_texts, val_labels)