import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SentimentClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', output_size=1):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

def preprocess_data(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    return encoding['input_ids'], encoding['attention_mask']

def train_model(model, data, labels, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        input_ids, attention_mask = preprocess_data(data)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

sentences = ['I love this product!', 'This is the worst experience.']
labels = torch.tensor([1, 0])
model = SentimentClassifier()
train_model(model, sentences, labels)