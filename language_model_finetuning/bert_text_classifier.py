import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return { 'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label, dtype=torch.long) }

def train_model(model, dataloader, optimizer, device, epochs):
    """Train the model with given parameters."""
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
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
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

def main():
    """Main function to execute training process."""
    texts = ["I love programming.", "Deep learning is fascinating.", "Python is great for data science.", "NLP is a subfield of AI."]
    labels = [1, 1, 1, 0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 20
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_dataset = TextDataset(texts_train, labels_train, tokenizer, max_length)
    val_dataset = TextDataset(texts_val, labels_val, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, optimizer, device, epochs=3)

if __name__ == '__main__':
    main()