import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

class TopicModel(nn.Module):
    def __init__(self):
        super(TopicModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 20)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

def preprocess_data():
    data = fetch_20newsgroups(subset='all')
    return data.data, data.target

def tokenize_data(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

def train_model(model, data_loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    texts, labels = preprocess_data()
    input_ids, attention_mask = tokenize_data(texts)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2)
    train_data = torch.utils.data.TensorDataset(train_inputs, train_labels)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8)
    model = TopicModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, optimizer, criterion)