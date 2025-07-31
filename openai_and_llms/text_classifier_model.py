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
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def preprocess(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, train_data, train_labels, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            inputs = self.preprocess(train_data)
            labels = torch.tensor(train_labels)
            self.optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess(texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            return predictions.tolist()

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='train', categories=['sci.space', 'comp.graphics'])
    train_data, val_data, train_labels, val_labels = train_test_split(data.data, data.target, test_size=0.2)
    classifier = TextClassifier(num_labels=2)
    classifier.train(train_data, train_labels)
    predictions = classifier.predict(val_data)
    print(predictions)