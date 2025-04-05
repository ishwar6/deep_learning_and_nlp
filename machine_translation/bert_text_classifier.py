import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels)

    def train(self, texts, labels, epochs=3, batch_size=8):
        input_ids, attention_mask, labels = self.preprocess_data(texts, labels)
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    def evaluate(self, texts, labels):
        input_ids, attention_mask, labels = self.preprocess_data(texts, labels)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = np.argmax(outputs.logits.numpy(), axis=1)
        accuracy = accuracy_score(labels.numpy(), predictions)
        print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    texts = ['I love programming', 'I hate bugs', 'Deep learning is fascinating', 'I enjoy learning new things']
    labels = [1, 0, 1, 1]
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
    classifier = TextClassifier()
    classifier.train(train_texts, train_labels)
    classifier.evaluate(val_texts, val_labels)