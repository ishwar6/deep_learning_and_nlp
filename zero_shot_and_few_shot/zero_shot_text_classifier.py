import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

class ZeroShotClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def predict(self, texts):
        inputs = self.preprocess(texts)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=1)

    def evaluate(self, texts, labels):
        predictions = self.predict(texts)
        return accuracy_score(labels, predictions.numpy())

if __name__ == '__main__':
    classifier = ZeroShotClassifier()
    sample_texts = ['I love deep learning.', 'This is a bad product.']
    sample_labels = [1, 0]
    accuracy = classifier.evaluate(sample_texts, sample_labels)
    print(f'Accuracy: {accuracy:.2f}')