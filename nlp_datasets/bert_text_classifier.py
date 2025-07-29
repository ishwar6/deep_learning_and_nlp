import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    """Custom dataset for loading text data."""
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

def train_model(model, data_loader, optimizer, device):
    """Train the model using the provided data loader."""
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
    """Main function to execute training on a sample dataset."""
    texts = ["I love programming in Python!", "Deep learning is fascinating.", "Natural language processing is a key area of AI."]
    labels = [1, 1, 1]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 32
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    average_loss = train_model(model, train_loader, optimizer, device)
    print(f'Training loss: {average_loss}')

if __name__ == '__main__':
    main()