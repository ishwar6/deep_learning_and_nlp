import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

class TextClassifier:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128)
        return encodings, labels

    def create_dataloader(self, encodings, labels, batch_size=8):
        dataset = torch.utils.data.TensorDataset(torch.tensor(encodings['input_ids']),
                                                 torch.tensor(encodings['attention_mask']),
                                                 torch.tensor(labels))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def train(self, train_texts, train_labels, eval_texts, eval_labels):
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        eval_encodings, eval_labels = self.preprocess_data(eval_texts, eval_labels)
        train_loader = self.create_dataloader(train_encodings, train_labels)
        eval_loader = self.create_dataloader(eval_encodings, eval_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch'
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader,
            eval_dataset=eval_loader
        )
        trainer.train()
        print('Training complete.')

if __name__ == '__main__':
    texts = ['I love programming.', 'Deep learning is fascinating.', 'Natural language processing is crucial.']
    labels = [1, 1, 1]
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
    classifier = TextClassifier(model_name='bert-base-uncased', num_labels=2)
    classifier.train(train_texts, train_labels, eval_texts, eval_labels)