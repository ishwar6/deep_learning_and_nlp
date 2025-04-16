import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BertTextClassifier(nn.Module):
    """
    A simple BERT-based text classifier.
    """
    def __init__(self, num_classes):
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state[:, 0, :])
        return logits

def preprocess_data(texts, labels):
    """
    Tokenizes and prepares the input data for BERT.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return encodings, torch.tensor(labels)

def train_model(model, dataloader, optimizer, device):
    """
    Trains the BERT model on the given data.
    """
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Training loss: {loss.item()}')

texts = ['I love programming.', 'Deep learning is fascinating.']
labels = [1, 1]
encodings, labels_tensor = preprocess_data(texts, labels)
dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
model = BertTextClassifier(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train_model(model, dataloader, optimizer, device='cpu')