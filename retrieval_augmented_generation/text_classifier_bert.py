import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess(self, texts, max_length=128):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3):
        train_encodings = self.preprocess(train_texts)
        dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

    def predict(self, texts):
        encodings = self.preprocess(texts)
        with torch.no_grad():
            outputs = self.model(encodings['input_ids'], encodings['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.numpy()

if __name__ == '__main__':
    classifier = TextClassifier()
    sample_texts = ['I love programming!', 'I hate bugs.']
    sample_labels = [1, 0]
    classifier.train(sample_texts, sample_labels)
    predictions = classifier.predict(['Programming is fun.', 'Bugs are annoying.'])
    print(predictions)