import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TransferLearningModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)

    def tokenize_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            inputs = self.tokenize_data(train_texts)
            labels = torch.tensor(train_labels)
            self.optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, test_texts):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenize_data(test_texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.numpy()

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='all')
    texts = data.data
    labels = data.target
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    transfer_model = TransferLearningModel()
    transfer_model.train(train_texts, train_labels, epochs=3)
    predictions = transfer_model.evaluate(test_texts)
    print(predictions)