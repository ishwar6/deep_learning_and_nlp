import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return encodings, torch.tensor(labels)

    def train(self, train_texts, train_labels, epochs=3):
        self.model.train()
        encodings, labels = self.preprocess_data(train_texts, train_labels)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(**encodings, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.numpy()

if __name__ == '__main__':
    sample_texts = ['I love this!', 'This is terrible.']
    sample_labels = [1, 0]
    train_texts, val_texts, train_labels, val_labels = train_test_split(sample_texts, sample_labels, test_size=0.2)
    analyzer = SentimentAnalyzer()
    analyzer.train(train_texts, train_labels)
    predictions = analyzer.predict(val_texts)
    accuracy = accuracy_score(val_labels, predictions)
    print(f'Validation Accuracy: {accuracy}')