import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

class SentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
        self.loss_fn = nn.CrossEntropyLoss()

    def preprocess(self, texts, max_length=128):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    def train(self, texts, labels, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            inputs = self.preprocess(texts)
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, texts, labels):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess(texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = accuracy_score(labels, predictions.numpy())
            print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    sample_texts = ['I love this product!', 'This is the worst experience ever.']
    sample_labels = torch.tensor([1, 0])
    classifier = SentimentClassifier()
    classifier.train(sample_texts, sample_labels)
    classifier.evaluate(sample_texts, sample_labels)