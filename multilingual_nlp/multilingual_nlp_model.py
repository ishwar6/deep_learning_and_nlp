import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class MultilingualNLPModel:
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3, batch_size=8):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        for epoch in range(epochs):
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                inputs = self.preprocess_data(batch_texts)
                outputs = self.model(**inputs, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(f'Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_data(texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            return predictions.tolist()

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='all')
    texts = data.data[:100]
    labels = data.target[:100]
    model = MultilingualNLPModel()
    model.train(texts, labels)
    predictions = model.predict(texts)
    print(predictions)