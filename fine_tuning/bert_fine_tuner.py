import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class ModelFineTuner:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_data(self, texts, max_length=128):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    def create_data_loader(self, texts, labels, batch_size=16):
        inputs = self.tokenize_data(texts)
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, train_texts, train_labels, epochs=3, batch_size=16):
        train_loader = self.create_data_loader(train_texts, train_labels, batch_size)
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

if __name__ == '__main__':
    texts = ['I love programming.', 'Deep learning is fascinating!', 'Transformers are great for NLP.']
    labels = [1, 1, 1]
    fine_tuner = ModelFineTuner(model_name='bert-base-uncased', num_labels=2)
    fine_tuner.train(texts, labels, epochs=2)