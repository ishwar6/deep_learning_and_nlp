import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return { 'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0], 'labels': torch.tensor(self.labels[idx], dtype=torch.long) }

def load_data():
    texts = ["I love programming!", "I hate bugs!", "Deep learning is fascinating.", "I dislike errors."]
    labels = [1, 0, 1, 0]
    return train_test_split(texts, labels, test_size=0.25, random_state=42)

def train_model():
    train_texts, val_texts, train_labels, val_labels = load_data()
    train_dataset = SentimentDataset(train_texts, train_labels)
    val_dataset = SentimentDataset(val_texts, val_labels)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    print("Training complete.")

if __name__ == '__main__':
    train_model()