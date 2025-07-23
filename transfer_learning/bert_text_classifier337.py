import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128)
        return encodings, labels

    def train(self, train_texts, train_labels, epochs=3, batch_size=8):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        inputs, labels = self.preprocess_data(train_texts, train_labels)
        dataset = torch.utils.data.TensorDataset(torch.tensor(inputs['input_ids']), torch.tensor(labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch[0], labels=batch[1])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        with torch.no_grad():
            encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128)
            outputs = self.model(torch.tensor(encodings['input_ids']))
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.numpy()

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='train')
    texts, labels = newsgroups.data, newsgroups.target
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
    classifier = TextClassifier('bert-base-uncased', num_labels=20)
    classifier.train(train_texts, train_labels)
    predictions = classifier.predict(val_texts)
    print(predictions)