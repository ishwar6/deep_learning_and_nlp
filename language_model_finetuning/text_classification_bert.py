import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np

class TextClassificationModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        return encodings, labels

    def train(self, train_texts, train_labels, eval_texts, eval_labels):
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        eval_encodings, eval_labels = self.preprocess_data(eval_texts, eval_labels)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_labels))
        eval_dataset = torch.utils.data.TensorDataset(torch.tensor(eval_encodings['input_ids']), torch.tensor(eval_labels))

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
            eval_dataset=eval_dataset,
        )

        trainer.train()

if __name__ == '__main__':
    dataset = fetch_20newsgroups(subset='all')
    texts, labels = dataset.data, dataset.target
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    model = TextClassificationModel()
    model.train(train_texts, train_labels, eval_texts, eval_labels)
    print('Training complete')