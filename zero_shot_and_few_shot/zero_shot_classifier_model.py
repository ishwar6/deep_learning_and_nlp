import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np

class ZeroShotClassifier:
    def __init__(self, model_name, num_classes):
        """Initializes the ZeroShotClassifier with a pretrained model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def predict(self, texts):
        """Predicts class probabilities for the given texts."""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=1).numpy()

# Simulating a dataset
texts = ["I love programming in Python!", "The weather is nice today.", "Artificial Intelligence is the future."]
labels = [0, 1, 2]  # Mock labels for three classes

# Splitting the dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initializing the model
model_name = 'distilbert-base-uncased'
classifier = ZeroShotClassifier(model_name, num_classes=3)

# Making predictions
predictions = classifier.predict(test_texts)
print("Predicted probabilities:", predictions)