import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

class TextDataset(Dataset):
    """Custom Dataset for loading text data for machine translation."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        label = torch.tensor(self.labels[idx])
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), label

class TranslationModel(nn.Module):
    """BERT-based model for text classification"""
    def __init__(self):
        super(TranslationModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

def train_model(dataset):
    """Train the translation model on the provided dataset."""
    model = TranslationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model.train()
    for epoch in range(3):
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

if __name__ == '__main__':
    texts = ['Hello world', 'Machine translation is fascinating', 'Deep learning is powerful']
    labels = [0, 1, 1]
    train_texts, _, train_labels, _ = train_test_split(texts, labels, test_size=0.2, random_state=42)
    dataset = TextDataset(train_texts, train_labels)
    train_model(dataset)