import numpy as np
import torch
from torch import nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

class TopicModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        return encodings, labels

    def train(self, texts, labels):
        encodings, labels = self.preprocess_data(texts, labels)
        dataset = TopicDataset(encodings, labels)
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

class TopicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    newsgroups = fetch_20newsgroups(subset='all')
    texts, labels = newsgroups.data, newsgroups.target
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2)
    model = TopicModel()
    model.train(texts_train, labels_train)
    print('Training complete.')

if __name__ == '__main__':
    main()