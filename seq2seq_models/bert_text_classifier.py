import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

class TextDataset(Dataset):
    """Custom dataset for loading text data and labels."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def preprocess_data(texts, labels):
    """Tokenizes text data using BERT tokenizer and splits into train/test sets."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return train_test_split(tokenized_texts['input_ids'], labels, test_size=0.2)

def train_model(train_data, train_labels):
    """Trains a BERT model on the provided training data and labels."""
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    dataset = TextDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model.train()
    for epoch in range(3):
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return model

if __name__ == '__main__':
    texts = ['I love programming.', 'Deep learning is fascinating.', 'Natural language processing is essential.']
    labels = [1, 1, 1]
    train_data, test_data, train_labels, test_labels = preprocess_data(texts, labels)
    model = train_model(train_data, train_labels)
    print('Training complete.')