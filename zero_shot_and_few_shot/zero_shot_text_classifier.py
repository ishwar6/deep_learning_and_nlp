import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ZeroShotClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def preprocess(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def predict(self, texts):
        inputs = self.preprocess(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).numpy()

if __name__ == '__main__':
    texts = ["The movie was fantastic!", "I did not enjoy the play.", "What a great film!"]
    labels = [2, 0, 2]  # 0: Negative, 1: Neutral, 2: Positive
    classifier = ZeroShotClassifier()
    predictions = classifier.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    print(f'Predicted labels: {predictions}')
    print(f'Accuracy: {accuracy:.2f}')