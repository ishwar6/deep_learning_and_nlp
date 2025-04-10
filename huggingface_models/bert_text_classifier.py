import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3, batch_size=8):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        inputs = self.preprocess_data(texts)
        for epoch in range(epochs):
            for i in range(0, len(texts), batch_size):
                optimizer.zero_grad()
                batch_inputs = {key: val[i:i + batch_size] for key, val in inputs.items()}
                outputs = self.model(**batch_inputs, labels=batch_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                print(f'Epoch {epoch + 1}, Batch {i // batch_size + 1}, Loss: {loss.item()}')

    def evaluate(self, texts):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_data(texts)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            return predictions.tolist()

if __name__ == '__main__':
    data = fetch_20newsgroups(subset='train')
    classifier = TextClassifier()
    classifier.train(data.data, data.target)
    sample_texts = data.data[:5]
    predictions = classifier.evaluate(sample_texts)
    print('Predictions for sample texts:', predictions)