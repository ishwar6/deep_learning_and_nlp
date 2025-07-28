import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class FineTuner:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        return tokens, torch.tensor(labels)

    def train(self, texts, labels):
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)
        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy='epoch',
            logging_dir='./logs'
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        trainer.train()
        return trainer

if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    num_labels = 2
    fine_tuner = FineTuner(model_name, num_labels)
    texts = ['I love programming.', 'Python is great!', 'I hate bugs.']
    labels = [1, 1, 0]
    trainer = fine_tuner.train(texts, labels)
    print('Training completed! Model trained successfully.')