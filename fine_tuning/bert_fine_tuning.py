import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def load_data():
    """
    Load and preprocess the 20 Newsgroups dataset.
    """
    newsgroups = fetch_20newsgroups(subset='train')
    return newsgroups.data, newsgroups.target


def tokenize_data(sentences):
    """
    Tokenize the input sentences using BERT's tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


def train_model(train_texts, train_labels):
    """
    Train a BERT model for sequence classification.
    """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)
    tokens = tokenize_data(train_texts)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=8,
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=torch.utils.data.TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(train_labels))
    )
    trainer.train()
    return model


def main():
    """
    Main function to execute fine-tuning on BERT model.
    """
    texts, labels = load_data()
    model = train_model(texts, labels)
    print('Model trained successfully!')


if __name__ == '__main__':
    main()