import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    df = pd.read_csv(file_path)
    return train_test_split(df['text'].values, df['label'].values, test_size=0.2)

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_texts, val_texts, train_labels, val_labels = load_data('sentiment_data.csv')
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len=128)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_len=128)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    print('Training completed.')

if __name__ == '__main__':
    main()