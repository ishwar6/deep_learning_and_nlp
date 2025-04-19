import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class SimpleTextDataset(Dataset):
    """A simple dataset for text inputs."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', padding=True, truncation=True)
        return encoding['input_ids'].squeeze(0), self.labels[idx]

class TextClassifier(nn.Module):
    """A simple BERT-based text classifier."""
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        return self.fc(outputs.pooler_output)

def train_model(model, dataloader, criterion, optimizer, epochs=3):
    """Train the text classification model."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')

texts = ['I love programming.', 'Deep learning is fascinating.']
labels = torch.tensor([1, 1])
train_dataset = SimpleTextDataset(texts, labels)
dataloader = DataLoader(train_dataset, batch_size=2)
model = TextClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
train_model(model, dataloader, criterion, optimizer)