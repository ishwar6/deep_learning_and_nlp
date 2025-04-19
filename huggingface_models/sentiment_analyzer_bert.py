import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the sentiment analyzer with a BERT model and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def tokenize_data(self, texts, max_length=128):
        """Tokenizes input texts and returns input IDs and attention masks."""
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    def train(self, train_texts, train_labels):
        """Prepares and trains the model on the given texts and labels."""
        train_encodings = self.tokenize_data(train_texts)
        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        trainer.train()

    def predict(self, texts):
        """Makes predictions for the input texts and returns class probabilities."""
        with torch.no_grad():
            encodings = self.tokenize_data(texts)
            outputs = self.model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
            return torch.softmax(outputs.logits, dim=-1)

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    mock_texts = ['I love this!', 'This is terrible.']
    mock_labels = [1, 0]
    analyzer.train(mock_texts, mock_labels)
    predictions = analyzer.predict(mock_texts)
    print(predictions)