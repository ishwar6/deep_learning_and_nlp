import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SentimentAnalyzer(nn.Module):
    """
    A simple sentiment analysis model using BERT.
    """
    def __init__(self):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data(texts):
    """
    Tokenizes and prepares input data for BERT.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    return encoding['input_ids'], encoding['attention_mask']

def train_model(model, data, labels, epochs=3):
    """
    Trains the sentiment analysis model.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        input_ids, attention_mask = data
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    texts = ['I love this!', 'This is terrible.']
    labels = torch.tensor([1, 0])
    input_ids, attention_mask = preprocess_data(texts)
    model = SentimentAnalyzer()
    train_model(model, (input_ids, attention_mask), labels)