import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], return_tensors='pt', padding=True, truncation=True)
        return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(), 'labels': torch.tensor(self.labels[idx])}

class SentimentAnalyzer:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def train(self, dataloader, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, text):
        self.model.eval()
        inputs = self.model.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1)
        return prediction.item()

if __name__ == '__main__':
    texts = ['I love this product!', 'This is the worst experience ever.']
    labels = [1, 0]
    dataset = SentimentDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.train(dataloader)
    test_text = 'I am so happy!'
    print(f'Test prediction: {sentiment_analyzer.predict(test_text)}')