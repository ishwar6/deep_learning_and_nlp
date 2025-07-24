import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels, max_length=128):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        return encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels)

    def train(self, texts, labels, epochs=3, batch_size=8):
        input_ids, attention_mask, labels_tensor = self.preprocess_data(texts, labels)
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()

        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, texts, labels):
        input_ids, attention_mask, labels_tensor = self.preprocess_data(texts, labels)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        accuracy = accuracy_score(labels_tensor.numpy(), predictions.numpy())
        print(f'Accuracy: {accuracy * 100:.2f}%')

# Mock data
texts = ['I love this!', 'This is terrible.', 'Absolutely fantastic!', 'Not good at all.']
labels = [1, 0, 1, 0]

analyzer = SentimentAnalyzer()
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
analyzer.train(train_texts, train_labels)
analyzer.evaluate(test_texts, test_labels)