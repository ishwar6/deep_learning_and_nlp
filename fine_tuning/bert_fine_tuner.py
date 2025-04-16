import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

class SimpleTextDataset(Dataset):
    """Custom dataset class for text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, return_tensors='pt')
        return { 'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[idx], dtype=torch.long) }

def train_model(model, dataset, epochs=3, batch_size=16):
    """Trains the BERT model on the given dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

if __name__ == '__main__':
    texts = ['I love programming.', 'Deep learning is fascinating.', 'Python is my favorite language.']
    labels = [1, 1, 1]
    dataset = SimpleTextDataset(texts, labels)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    train_model(model, dataset)