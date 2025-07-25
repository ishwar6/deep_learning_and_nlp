import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3, batch_size=8):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        inputs = self.preprocess_data(texts)
        dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch[0], attention_mask=batch[1], labels=batch[2])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        inputs = self.preprocess_data(texts)
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='train')
    texts, labels = data.data, data.target
    classifier = TextClassifier(num_labels=20)
    classifier.train(texts, labels)
    sample_texts = texts[:5]
    predictions = classifier.predict(sample_texts)
    print(f'Sample Predictions: {predictions}')