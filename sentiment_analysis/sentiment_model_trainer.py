import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis using text data."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), label

class SentimentAnalyzer:
    """Sentiment analysis model using BERT architecture."""
    def __init__(self, num_classes):
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def train(self, dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, text):
        self.model.eval()
        inputs = self.model.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

if __name__ == '__main__':
    texts = ['I love this!', 'This is awful.', 'Absolutely fantastic!']
    labels = [1, 0, 1]
    dataset = SentimentDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    sentiment_analyzer = SentimentAnalyzer(num_classes=2)
    sentiment_analyzer.train(dataloader, epochs=3)
    print(sentiment_analyzer.predict('I hate this!'))