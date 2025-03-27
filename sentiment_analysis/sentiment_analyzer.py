import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def preprocess_data(self, texts, labels, test_size=0.2):
        tokenized_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        return train_test_split(input_ids, attention_mask, labels, test_size=test_size)

    def train(self, train_inputs, train_masks, train_labels, epochs=3, batch_size=16):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_masks, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                b_input_ids, b_input_mask, b_labels = [item.to(self.device) for item in batch]
                self.model.zero_grad()
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

    def predict(self, texts):
        self.model.eval()
        with torch.no_grad():
            tokenized_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            input_ids = tokenized_inputs['input_ids'].to(self.device)
            attention_mask = tokenized_inputs['attention_mask'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            return predictions.cpu().numpy()

if __name__ == '__main__':
    dataset = load_dataset('imdb')
    texts = dataset['train']['text'][:100]
    labels = dataset['train']['label'][:100]
    analyzer = SentimentAnalyzer()
    train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = analyzer.preprocess_data(texts, labels)
    analyzer.train(train_inputs, train_masks, train_labels)
    predictions = analyzer.predict(test_inputs)
    print(predictions)