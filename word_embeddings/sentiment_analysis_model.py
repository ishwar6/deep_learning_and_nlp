import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class SimpleDataset(Dataset):
    """Custom dataset for loading text data and labels."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', padding=True, truncation=True)
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), self.labels[idx]

class SentimentModel(nn.Module):
    """BERT-based sentiment analysis model."""
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def train_model(model, dataloader, epochs=3):
    """Trains the sentiment model on the given dataset."""
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

if __name__ == '__main__':
    texts = ['I love this movie!', 'This was terrible.']
    labels = [1, 0]
    dataset = SimpleDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = SentimentModel()
    train_model(model, dataloader)