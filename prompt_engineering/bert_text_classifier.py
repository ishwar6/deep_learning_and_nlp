import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)

    def tokenize_data(self, texts, max_length=128):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3, batch_size=16):
        inputs = self.tokenize_data(train_texts)
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(train_labels))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

    def predict(self, texts):
        self.model.eval()
        inputs = self.tokenize_data(texts)
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()

if __name__ == '__main__':
    texts = ['I love programming!', 'I hate bugs.']
    labels = [1, 0]
    classifier = TextClassifier()
    classifier.train(texts, labels)
    predictions = classifier.predict(['I enjoy coding.', 'I dislike errors.'])
    print(predictions)