import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        """Initializes the sentiment analyzer with a pre-trained BERT model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def encode_data(self, texts, labels):
        """Encodes the text data and returns inputs and labels for training."""
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        return encodings, torch.tensor(labels)

    def train(self, texts, labels):
        """Trains the model on the provided texts and labels."""
        encodings, labels = self.encode_data(texts, labels)
        dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir='./logs'
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()

    def predict(self, texts):
        """Makes predictions on new texts."""
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.tolist()

if __name__ == '__main__':
    texts = ['I love this product!', 'This is the worst thing ever.']
    labels = [1, 0]
    analyzer = SentimentAnalyzer()
    analyzer.train(texts, labels)
    predictions = analyzer.predict(['I am so happy!', 'I dislike this.'])
    print(predictions)