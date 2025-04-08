import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    """A custom dataset for loading text data for NLP tasks."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train(model, dataloader, optimizer, device):
    """Training loop for the model."""
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def main():
    """Main function to train the BERT model on sample data."""
    texts = ['Hello world', 'Deep learning is amazing', 'Natural language processing with transformers']
    labels = [0, 1, 1]
    dataset = SimpleDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss = train(model, dataloader, optimizer, device)
    print(f'Training loss: {loss:.4f}')  

if __name__ == '__main__':
    main()