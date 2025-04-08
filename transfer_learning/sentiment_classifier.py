import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class SentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize_data(self, texts, max_length=128):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    def train(self, train_texts, train_labels, epochs=3):
        train_encodings = self.tokenize_data(train_texts)
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

    def predict(self, texts):
        encodings = self.tokenize_data(texts)
        with torch.no_grad():
            outputs = self.model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()

if __name__ == '__main__':
    classifier = SentimentClassifier()
    sample_texts = ["I love this movie!", "This was the worst experience ever."]
    sample_labels = [1, 0]
    classifier.train(sample_texts, sample_labels)
    predictions = classifier.predict(sample_texts)
    print(predictions)