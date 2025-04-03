import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.datasets import fetch_20newsgroups

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_classes=20):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def preprocess_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, train_texts, train_labels):
        inputs = self.preprocess_data(train_texts)
        dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(train_labels))
        train_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=dataset,
        )
        trainer.train()

    def predict(self, texts):
        inputs = self.preprocess_data(texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).tolist()

if __name__ == '__main__':
    newsgroups = fetch_20newsgroups(subset='train')
    classifier = TextClassifier()
    classifier.train(newsgroups.data, newsgroups.target)
    test_texts = ['This is a technology article.', 'A discussion about politics.']
    predictions = classifier.predict(test_texts)
    print(predictions)