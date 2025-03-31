import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class SentimentAnalysisModel:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128)
        return encodings, labels

    def train(self, texts, labels):
        encodings, labels = self.preprocess_data(texts, labels)
        dataset = {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': labels}
        train_dataset, eval_dataset = train_test_split(dataset, test_size=0.1)
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        print('Training completed!')

    def predict(self, texts):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.numpy()

if __name__ == '__main__':
    model = SentimentAnalysisModel()
    sample_texts = ['I love this!', 'This is terrible.']
    sample_labels = [1, 0]
    model.train(sample_texts, sample_labels)
    predictions = model.predict(['What a great day!', 'I am not happy at all.'])
    print('Predictions:', predictions)