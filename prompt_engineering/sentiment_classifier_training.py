import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return self.sigmoid(logits)

def preprocess_data(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

def train_model(model, data_loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    mock_texts = ['I love this!', 'This is terrible.']
    mock_labels = torch.tensor([1, 0])
    processed_data = preprocess_data(mock_texts)
    model = SentimentClassifier()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.BCELoss()
    train_model(model, [(processed_data, mock_labels)], optimizer, criterion)