import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import numpy as np

class SentimentClassifier(nn.Module):
    """
    A simple sentiment classifier using BERT embeddings.
    """
    def __init__(self, num_classes=2):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def prepare_data(samples, tokenizer, max_length=128):
    """
    Prepares input data for the model.
    """
    input_ids, attention_masks = [], []
    for sample in samples:
        encoded = tokenizer.encode_plus(
            sample,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return torch.cat(input_ids), torch.cat(attention_masks)

def train_model(model, train_dataloader, optimizer, device):
    """
    Trains the sentiment classifier.
    """
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dataloader)

if __name__ == '__main__':
    samples = ['I love this!', 'This is terrible.', 'Absolutely fantastic!', 'Not good at all.']
    labels = [1, 0, 1, 0]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_masks = prepare_data(samples, tokenizer)
    X_train, X_val, y_train, y_val = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
    train_data = torch.utils.data.TensorDataset(X_train, attention_masks[:len(X_train)], torch.tensor(y_train))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2)
    model = SentimentClassifier().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss = train_model(model, train_dataloader, optimizer, 'cuda')
    print(f'Training loss: {loss:.4f}')