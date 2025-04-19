import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TextClassifier:
    def __init__(self, num_labels):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

    def train(self, train_loader, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

texts = ['I love programming', 'Deep learning is fascinating', 'NLP is an interesting field']
labels = [1, 1, 1]
train_dataset = TextDataset(texts, labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
classifier = TextClassifier(num_labels=2)
classifier.train(train_loader)