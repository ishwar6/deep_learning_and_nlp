import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class SimpleDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, return_tensors='pt')
        return { 'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[idx], dtype=torch.long) }

class BertClassifier:
    def __init__(self, num_classes):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    def train(self, dataloader, epochs=3, lr=5e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

texts = ['I love programming.', 'BERT is a great model for NLP.', 'Deep learning is fascinating.']
labels = [1, 1, 1]

dataset = SimpleDataset(texts, labels)
loader = DataLoader(dataset, batch_size=2)
classifier = BertClassifier(num_classes=2)
classifier.train(loader)