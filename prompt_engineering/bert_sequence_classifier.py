import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def load_data():
    """Loads and preprocesses the 20 Newsgroups dataset."""
    newsgroups = fetch_20newsgroups(subset='all')
    return newsgroups.data, newsgroups.target


def tokenize_data(texts):
    """Tokenizes input texts using BERT tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


def create_dataloaders(texts, labels, batch_size=16):
    """Creates PyTorch dataloaders for the dataset."""
    dataset = torch.utils.data.TensorDataset(texts['input_ids'], texts['attention_mask'], torch.tensor(labels))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(train_loader, val_loader, epochs=3):
    """Trains the BERT sequence classification model."""
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs} completed. Loss: {loss.item()}')


def main():
    """Main function to execute the workflow."""
    texts, labels = load_data()
    tokenized = tokenize_data(texts)
    train_loader, val_loader = create_dataloaders(tokenized, labels)
    train_model(train_loader, val_loader)


if __name__ == '__main__':
    main()