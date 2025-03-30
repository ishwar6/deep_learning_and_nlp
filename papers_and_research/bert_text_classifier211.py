import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class TextDataset(Dataset):
    """Custom dataset for text data using BERT tokenizer."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), self.labels[idx]

class SimpleClassifier(nn.Module):
    """A simple classifier using BERT embeddings."""
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 2)  # Assuming binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def train_model(dataset, epochs=3, batch_size=8):
    """Train the BERT classifier on the given dataset."""
    model = SimpleClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}')

if __name__ == '__main__':
    sample_texts = ['I love deep learning', 'Natural Language Processing is fascinating']
    sample_labels = [1, 0]
    dataset = TextDataset(sample_texts, sample_labels)
    train_model(dataset)