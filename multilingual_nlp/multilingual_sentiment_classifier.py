import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np

class MultilingualSentimentClassifier:
    """
    A simple multilingual sentiment classifier using BERT.
    """
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

    def preprocess_data(self, texts, labels):
        """
        Tokenizes and encodes the input texts.
        """
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return encodings, torch.tensor(labels)

    def train(self, texts, labels, epochs=3):
        """
        Trains the BERT model on the provided texts and labels.
        """
        self.model.train()
        encodings, labels_tensor = self.preprocess_data(texts, labels)
        for epoch in range(epochs):
            outputs = self.model(**encodings, labels=labels_tensor)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, texts):
        """
        Predicts sentiment for the input texts.
        """
        self.model.eval()
        with torch.no_grad():
            encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
            return predictions.tolist()

if __name__ == '__main__':
    texts = ['I love this product!', 'This is the worst experience ever.']
    labels = [1, 0]
    classifier = MultilingualSentimentClassifier()
    classifier.train(texts, labels)
    predictions = classifier.predict(texts)
    print('Predictions:', predictions)