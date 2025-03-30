import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)

    def preprocess(self, texts, max_length=128):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        return inputs['input_ids'], inputs['attention_mask']

    def train(self, texts, labels, epochs=3, batch_size=8):
        self.model.train()
        input_ids, attention_mask = self.preprocess(texts)
        dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, torch.tensor(labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, texts, labels):
        self.model.eval()
        with torch.no_grad():
            input_ids, attention_mask = self.preprocess(texts)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            print(classification_report(labels, predictions.cpu().numpy()))

if __name__ == '__main__':
    texts = ['I love programming.', 'Python is great!', 'I hate bugs.']
    labels = [1, 1, 0]
    classifier = TextClassifier('bert-base-uncased', num_labels=2)
    classifier.train(texts, labels)
    classifier.evaluate(texts, labels)