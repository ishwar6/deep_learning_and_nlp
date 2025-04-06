import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

class SentimentDataset(Dataset):
    """Custom Dataset for loading sentiment data."""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
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

def train_model(model, data_loader, optimizer, device):
    """Train the model for one epoch."""
    model = model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

def main():
    """Main function to execute model training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    texts = ["I love this product!", "This is the worst service ever.", "Absolutely fantastic!", "Not worth the money."]
    labels = [1, 0, 1, 0]
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2)
    train_dataset = SentimentDataset(texts_train, labels_train, tokenizer, max_length=20)
    val_dataset = SentimentDataset(texts_val, labels_val, tokenizer, max_length=20)
    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    for epoch in range(3):
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f'Epoch {epoch + 1}, Training Loss: {train_loss}')

if __name__ == '__main__':
    main()