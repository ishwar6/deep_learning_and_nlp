import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class MultilingualNLPModel:
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3, batch_size=8):
        self.model.train()
        train_encodings = self.tokenize_data(train_texts)
        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    def evaluate(self, test_texts, test_labels):
        self.model.eval()
        with torch.no_grad():
            test_encodings = self.tokenize_data(test_texts)
            outputs = self.model(test_encodings['input_ids'], attention_mask=test_encodings['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == torch.tensor(test_labels)).float().mean().item()
            print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='all')
    texts = newsgroups.data
    labels = newsgroups.target
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    model = MultilingualNLPModel()
    model.train(train_texts, train_labels)
    model.evaluate(test_texts, test_labels)