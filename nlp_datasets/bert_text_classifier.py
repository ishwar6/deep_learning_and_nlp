import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

class TextDataset(Dataset):
    """
    Custom dataset for text classification.
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(train_texts, train_labels, epochs=3, batch_size=16):
    """
    Train a BERT model on the given texts and labels.
    """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(train_labels)))
    dataset = TextDataset(train_texts, train_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

if __name__ == '__main__':
    texts = ["I love programming!", "Deep learning is fascinating.", "Python is my favorite language.", "NLP is amazing.", "Artificial Intelligence is the future."]
    labels = [1, 1, 1, 1, 0]
    train_texts, _, train_labels, _ = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_model(train_texts, train_labels)