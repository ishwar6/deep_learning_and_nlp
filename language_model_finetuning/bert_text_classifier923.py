import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_data(self, texts, max_length=128):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    def prepare_data(self, texts, labels, test_size=0.2):
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=test_size)
        train_encodings = self.tokenize_data(train_texts)
        val_encodings = self.tokenize_data(val_texts)
        return train_encodings, train_labels, val_encodings, val_labels

    def train(self, train_encodings, train_labels, val_encodings, val_labels):
        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
        val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        trainer.train()

    def evaluate(self, texts, labels):
        encodings = self.tokenize_data(texts)
        inputs = encodings['input_ids'], encodings['attention_mask']
        with torch.no_grad():
            logits = self.model(*inputs).logits
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions.numpy() == labels).mean()
        print(f'Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    example_texts = ['I love programming.', 'I hate bugs.', 'Python is great.']
    example_labels = [1, 0, 1]
    classifier = TextClassifier('bert-base-uncased', num_labels=2)
    train_encodings, train_labels, val_encodings, val_labels = classifier.prepare_data(example_texts, example_labels)
    classifier.train(train_encodings, train_labels, val_encodings, val_labels)
    classifier.evaluate(example_texts, example_labels)