import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class TextDataset(Dataset):
    """Custom dataset for loading text data and creating tokenized input for BERT."""
    def __init__(self, texts, labels, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

texts = ["Hello, how are you?", "I am fine, thank you!"]
labels = [0, 1]

dataset = TextDataset(texts, labels)

def create_data_loader(dataset, batch_size):
    """Creates a DataLoader for the dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

data_loader = create_data_loader(dataset, batch_size=2)

for batch in data_loader:
    print(batch['input_ids'], batch['attention_mask'], batch['label'])