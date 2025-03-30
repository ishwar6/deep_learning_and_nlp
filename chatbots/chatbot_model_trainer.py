import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class ChatbotModel:
    def __init__(self, model_name='bert-base-uncased', num_classes=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3, lr=1e-5):
        inputs = self.preprocess_data(texts)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_data(texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='train')
    texts, labels = data.data, data.target
    model = ChatbotModel()
    model.train(texts, labels)
    test_texts = texts[:5]
    predictions = model.predict(test_texts)
    print('Predictions for test texts:', predictions)