import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis using BERT tokenizer."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
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

def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    """Creates a DataLoader for the dataset."""
    dataset = SentimentDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    """Trains the model for one epoch."""
    model = model.train()
    total_loss = 0
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()
        correct_predictions += (outputs.logits.argmax(dim=1) == labels).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader), correct_predictions / len(data_loader.dataset)

def main():
    """Main function to run the sentiment analysis training loop."""
    texts = ["I love this product!", "This is the worst experience ever.", "Absolutely fantastic!", "I am not happy with this."]
    labels = [1, 0, 1, 0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_data_loader = create_data_loader(texts, labels, tokenizer, max_len=32, batch_size=2)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_data_loader, loss_fn, optimizer, device)
        print(f'Epoch {epoch + 1}, Loss: {train_loss}, Accuracy: {train_acc}')

if __name__ == '__main__':
    main()