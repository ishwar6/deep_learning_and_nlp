import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset

class LanguageModelFinetuner:
    def __init__(self, model_name, num_labels):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        return encodings, labels

    def train(self, texts, labels, epochs=3, batch_size=8):
        encodings, labels = self.preprocess_data(texts, labels)
        dataset = { 'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': labels }
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(dataset['input_ids']), torch.tensor(dataset['attention_mask']), torch.tensor(dataset['labels']))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)

        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, texts, labels):
        encodings, labels = self.preprocess_data(texts, labels)
        with torch.no_grad():
            outputs = self.model(torch.tensor(encodings['input_ids']), attention_mask=torch.tensor(encodings['attention_mask']))
            predictions = outputs.logits.argmax(dim=-1)
        accuracy = (predictions.numpy() == labels).mean()
        print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    dataset = load_dataset('glue', 'mrpc')
    texts = dataset['train']['sentence1'] + dataset['train']['sentence2']
    labels = dataset['train']['label']
    finetuner = LanguageModelFinetuner('bert-base-uncased', num_labels=2)
    finetuner.train(texts, labels)
    finetuner.evaluate(texts, labels)