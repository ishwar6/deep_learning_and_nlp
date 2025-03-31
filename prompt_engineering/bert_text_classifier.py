import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def encode_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            inputs = self.encode_data(texts)
            outputs = self.model(**inputs, labels=torch.tensor(labels))
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

    def predict(self, texts):
        self.model.eval()
        inputs = self.encode_data(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.numpy()

if __name__ == '__main__':
    sample_texts = ['I love programming.', 'I hate bugs.']
    sample_labels = [1, 0]
    classifier = TextClassifier(num_classes=2)
    classifier.train(sample_texts, sample_labels, epochs=2)
    predictions = classifier.predict(['Programming is fun.'])
    print('Predictions:', predictions)