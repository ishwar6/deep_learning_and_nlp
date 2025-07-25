import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    """Custom dataset for text classification using BERT."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', padding=True, truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[idx])}

class TextClassifier:
    """Text classification model based on BERT."""
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def train(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        return total_loss / len(dataloader)

texts = ['This is a positive example.', 'This is a negative example.']
labels = [1, 0]

dataset = SimpleDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=2)
classifier = TextClassifier()
loss = classifier.train(dataloader)
print(f'Training loss: {loss:.4f}')