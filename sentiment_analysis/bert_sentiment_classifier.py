import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs['labels'] = torch.tensor(labels)
        return inputs

    def train(self, texts, labels, epochs=3, batch_size=8):
        inputs = self.preprocess_data(texts, labels)
        train_size = int(0.8 * len(inputs['input_ids']))
        train_dataset = {key: val[:train_size] for key, val in inputs.items()}
        eval_dataset = {key: val[train_size:] for key, val in inputs.items()}
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            evaluation_strategy='epoch',
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()

if __name__ == '__main__':
    texts = ['I love this product!', 'This is the worst thing I have ever bought.']
    labels = [1, 0]
    analyzer = SentimentAnalyzer()
    analyzer.train(texts, labels)
    predictions = analyzer.predict(texts)
    print(predictions)