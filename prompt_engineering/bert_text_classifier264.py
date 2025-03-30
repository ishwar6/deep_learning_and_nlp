import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        return encodings, labels

    def train(self, train_texts, train_labels, epochs=3):
        self.model.train()
        encodings, labels = self.preprocess_data(train_texts, train_labels)
        dataset = torch.utils.data.TensorDataset(torch.tensor(encodings['input_ids']), 
                                                 torch.tensor(encodings['attention_mask']), 
                                                 torch.tensor(labels))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        for epoch in range(epochs):
            for batch in train_loader:
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, test_texts, test_labels):
        self.model.eval()
        with torch.no_grad():
            encodings, labels = self.preprocess_data(test_texts, test_labels)
            outputs = self.model(torch.tensor(encodings['input_ids']), 
                                 attention_mask=torch.tensor(encodings['attention_mask']))
            predictions = torch.argmax(outputs.logits, dim=1)
            accuracy = accuracy_score(labels, predictions.numpy())
            print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    texts = ['I love programming!', 'Python is great for data science.', 'I dislike bugs in code.']
    labels = [1, 1, 0]
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    classifier = TextClassifier(model_name='bert-base-uncased', num_labels=2)
    classifier.train(train_texts, train_labels, epochs=2)
    classifier.evaluate(test_texts, test_labels)