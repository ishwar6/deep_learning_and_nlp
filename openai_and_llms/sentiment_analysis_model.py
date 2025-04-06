import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

class SentimentAnalysisModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_data(self, texts, max_length=128):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    def train(self, texts, labels, epochs=3, batch_size=8):
        inputs = self.tokenize_data(texts)
        dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
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

    def predict(self, texts):
        inputs = self.tokenize_data(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=-1).numpy()

if __name__ == '__main__':
    texts = ['I love this product!', 'This is the worst experience I have ever had.']
    labels = [1, 0]
    model = SentimentAnalysisModel()
    model.train(texts, labels)
    predictions = model.predict(['This was fantastic!', 'I do not like this.'])
    print(predictions)