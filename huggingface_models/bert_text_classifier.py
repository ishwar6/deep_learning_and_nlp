import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW

class TextDataset(Dataset):
    """Custom Dataset for loading text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', padding=True, truncation=True, max_length=128)
        return { 'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0], 'labels': torch.tensor(self.labels[idx], dtype=torch.long) }

def train_model(dataset, batch_size=8, epochs=3):
    """Function to train the BERT model on the given dataset."""
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

if __name__ == '__main__':
    texts = ['I love programming.', 'Deep learning is fascinating!', 'Hugging Face makes NLP easy.']
    labels = [1, 1, 1]
    dataset = TextDataset(texts, labels)
    train_model(dataset)