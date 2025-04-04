import spacy
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

    def tokenize_data(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    def train(self, texts, labels, epochs=3, batch_size=8):
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
        train_encodings = self.tokenize_data(train_texts)
        val_encodings = self.tokenize_data(val_texts)

        train_labels_tensor = torch.tensor(train_labels)
        val_labels_tensor = torch.tensor(val_labels)

        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels_tensor)
        val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels_tensor)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()

        for epoch in range(epochs):
            for i in range(0, len(train_dataset), batch_size):
                batch = train_dataset[i:i + batch_size]
                optimizer.zero_grad()
                outputs = self.model(batch[0], attention_mask=batch[1], labels=batch[2])
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        self.evaluate(val_dataset)

    def evaluate(self, val_dataset):
        self.model.eval()
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(batch[0], attention_mask=batch[1])
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.numpy())
                all_labels.extend(batch[2].numpy())

        print(classification_report(all_labels, all_preds))

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    texts = ["This is a great movie!", "I did not like this film.", "Absolutely fantastic!", "Terrible experience."]
    labels = [1, 0, 1, 0]
    classifier = TextClassifier()
    classifier.train(texts, labels)