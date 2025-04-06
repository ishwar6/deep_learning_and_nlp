import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3, batch_size=8):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        inputs = self.preprocess_data(texts)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, texts, labels):
        self.model.eval()
        inputs = self.preprocess_data(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == labels).float().mean().item()
            print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    dataset = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'soc.religion.christian'], remove=('headers', 'footers', 'quotes'))
    texts = dataset.data
    labels = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    classifier = TextClassifier('distilbert-base-uncased', num_labels=2)
    classifier.train(X_train, torch.tensor(y_train))
    classifier.evaluate(X_test, torch.tensor(y_test))