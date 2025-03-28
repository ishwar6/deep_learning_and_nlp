import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class SentimentAnalysisModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        """Initializes the sentiment analysis model with BERT."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels, max_length=128):
        """Tokenizes the input texts and prepares them for training."""
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        return encodings, labels

    def train(self, texts, labels, epochs=3, batch_size=16):
        """Trains the sentiment analysis model on the provided texts and labels."""
        encodings, labels = self.preprocess_data(texts, labels)
        dataset = torch.utils.data.TensorDataset(torch.tensor(encodings['input_ids']), 
                                                 torch.tensor(encodings['attention_mask']), 
                                                 torch.tensor(labels))
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=10,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()

    def predict(self, texts):
        """Predicts the sentiment for a given list of texts."""
        self.model.eval()
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.numpy()  

if __name__ == '__main__':
    model = SentimentAnalysisModel()
    sample_texts = ['I love this!', 'This is terrible.']
    sample_labels = [1, 0]
    model.train(sample_texts, sample_labels)
    predictions = model.predict(sample_texts)
    print(predictions)