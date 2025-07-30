import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess(self, texts, labels):
        tokenized_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        return tokenized_inputs, torch.tensor(labels)

    def train(self, texts, labels):
        inputs, labels = self.preprocess(texts, labels)
        train_size = int(0.8 * len(texts))
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs['input_ids'], labels, train_size=train_size)
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=(train_inputs, train_labels),
            eval_dataset=(val_inputs, val_labels)
        )
        trainer.train()

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.numpy()

if __name__ == '__main__':
    classifier = TextClassifier()
    mock_texts = ['I love programming.', 'Python is great!', 'I dislike bugs.']
    mock_labels = [1, 1, 0]
    classifier.train(mock_texts, mock_labels)
    predictions = classifier.predict(mock_texts)
    print('Predictions:', predictions)