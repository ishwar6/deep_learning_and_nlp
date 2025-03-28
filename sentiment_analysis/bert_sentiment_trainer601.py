import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader, Dataset

class SentimentDataset(Dataset):
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

def train(model, dataloader, optimizer, device):
    model = model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataset = fetch_20newsgroups(subset='train', categories=['sci.electronics', 'rec.autos'], remove=('headers', 'footers', 'quotes'))
    texts, labels = dataset.data, dataset.target
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

    train_dataset = SentimentDataset(texts_train, labels_train, tokenizer, max_length=128)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        loss = train(model, train_dataloader, optimizer, device)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

if __name__ == '__main__':
    main()