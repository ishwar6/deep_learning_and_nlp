import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np

class FewShotClassifier:
    def __init__(self, model_name, num_classes):
        """Initialize the few-shot classifier with a pre-trained model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)

    def preprocess_data(self, texts, labels):
        """Tokenize and encode the input texts."""
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return encodings, torch.tensor(labels)

    def train(self, texts, labels, epochs=3):
        """Train the model on few-shot data."""
        encodings, labels = self.preprocess_data(texts, labels)
        dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, texts):
        """Make predictions on new texts."""
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.numpy()

if __name__ == '__main__':
    texts = ["I love programming.", "Python is great!", "Deep learning is fascinating."]
    labels = [0, 0, 1]
    model_name = 'distilbert-base-uncased'
    classifier = FewShotClassifier(model_name, num_classes=2)
    classifier.train(texts, labels, epochs=2)
    predictions = classifier.predict(["I enjoy coding.", "Neural networks are awesome."])
    print(f'Predictions: {predictions}')