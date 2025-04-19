import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

class TopicModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512)
        return encodings, labels

    def train(self, texts, labels, epochs=3):
        self.model.train()
        encodings, labels = self.preprocess_data(texts, labels)
        dataset = torch.utils.data.TensorDataset(torch.tensor(encodings['input_ids']),
                                                 torch.tensor(encodings['attention_mask']),
                                                 torch.tensor(labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        for epoch in range(epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

    def evaluate(self, texts, labels):
        self.model.eval()
        encodings, labels = self.preprocess_data(texts, labels)
        with torch.no_grad():
            outputs = self.model(torch.tensor(encodings['input_ids']),
                                 attention_mask=torch.tensor(encodings['attention_mask']))
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions.numpy() == labels).mean()
            print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='all')
    texts, labels = newsgroups.data, newsgroups.target
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    model = TopicModel()
    model.train(train_texts, train_labels)
    model.evaluate(test_texts, test_labels)