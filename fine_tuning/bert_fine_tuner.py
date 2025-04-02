import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class FineTuner:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        return encodings, labels

    def fine_tune(self, train_encodings, train_labels, val_encodings, val_labels):
        train_dataset = torch.utils.data.Dataset(train_encodings, train_labels)
        val_dataset = torch.utils.data.Dataset(val_encodings, val_labels)

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
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        return trainer

if __name__ == '__main__':
    dataset = load_dataset('imdb')
    train_texts, val_texts, train_labels, val_labels = train_test_split(dataset['train']['text'], dataset['train']['label'], test_size=0.2)
    ft = FineTuner('bert-base-uncased', num_labels=2)
    train_encodings, train_labels = ft.preprocess_data(train_texts, train_labels)
    val_encodings, val_labels = ft.preprocess_data(val_texts, val_labels)
    trainer = ft.fine_tune(train_encodings, train_labels, val_encodings, val_labels)
    print('Fine-tuning completed. Model saved.')