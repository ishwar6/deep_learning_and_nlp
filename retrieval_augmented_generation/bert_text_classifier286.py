import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset):
        return dataset.map(lambda x: self.tokenizer(x['text'], padding='max_length', truncation=True), batched=True)

    def train(self, train_dataset):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def predict(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.tolist()

if __name__ == '__main__':
    classifier = TextClassifier()
    dataset = load_dataset('imdb', split='train')
    processed_data = classifier.preprocess_data(dataset)
    classifier.train(processed_data)
    sample_texts = ['This movie was fantastic!', 'I did not like this film.']
    predictions = classifier.predict(sample_texts)
    print(predictions)