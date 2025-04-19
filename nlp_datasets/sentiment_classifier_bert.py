import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class TextDataset(Dataset):
    """Custom Dataset for loading text data."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    """Sentiment classification using BERT."""
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        drop_output = self.dropout(pooled_output)
        return self.linear(drop_output)

# Sample data
texts = ["I love programming.", "I hate bugs.", "Python is amazing!"]
labels = [1, 0, 1]

# Parameters
max_length = 10
batch_size = 2

# Load tokenizer and create dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2)
train_dataset = TextDataset(X_train, y_train, tokenizer, max_length)
val_dataset = TextDataset(X_val, y_val, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
model = SentimentClassifier(n_classes=2)
optimizer = optim.Adam(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(2):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')