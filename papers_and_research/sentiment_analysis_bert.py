import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

class TextDataset(Dataset):
    """Custom dataset for loading text and labels."""
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
    """BERT-based model for sentiment classification."""
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

def train_model(model, dataloader, epochs=3):
    """Train the sentiment classification model."""
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

if __name__ == '__main__':
    mock_texts = ["I love this product!", "This is the worst experience I have ever had."]
    mock_labels = [1, 0]
    dataset = TextDataset(mock_texts, mock_labels)
    dataloader = DataLoader(dataset, batch_size=2)
    model = SentimentModel()
    train_model(model, dataloader, epochs=2)