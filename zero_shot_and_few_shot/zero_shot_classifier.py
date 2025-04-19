import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import numpy as np

class ZeroShotClassifier:
    def __init__(self, model_name):
        """Initializes the ZeroShotClassifier with a specified model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, texts, labels):
        """Predicts the class labels for given texts using zero-shot classification."""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        return predictions.numpy(), probabilities.numpy()

    def evaluate(self, texts, true_labels, labels):
        """Evaluates the model's predictions against the true labels."""
        predicted_labels, _ = self.predict(texts, labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        return accuracy

if __name__ == '__main__':
    model_name = 'facebook/bart-large-mnli'
    classifier = ZeroShotClassifier(model_name)
    sample_texts = ['I love programming in Python.', 'The weather is great today.']
    sample_labels = ['programming', 'weather']
    true_labels = [0, 1]
    accuracy = classifier.evaluate(sample_texts, true_labels, sample_labels)
    print(f'Accuracy: {accuracy:.2f}')