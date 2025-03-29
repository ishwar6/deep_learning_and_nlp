import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess(self, texts, max_length=128):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3):
        train_encodings = self.preprocess(train_texts)
        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()

    def evaluate(self, texts):
        self.model.eval()
        encodings = self.preprocess(texts)
        with torch.no_grad():
            outputs = self.model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.numpy()

if __name__ == '__main__':
    texts = ['I love programming!', 'I hate bugs.']
    labels = [1, 0]
    classifier = TextClassifier()
    classifier.train(texts, labels)
    predictions = classifier.evaluate(['Programming is fun!', 'Bugs are annoying.'])
    print(predictions)