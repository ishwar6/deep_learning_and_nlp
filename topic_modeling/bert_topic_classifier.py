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
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data():
    data = fetch_20newsgroups(subset='all')
    return data.data, data.target

def tokenize_data(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

def train_model(model, inputs, masks, labels, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

def main():
    texts, labels = preprocess_data()
    input_ids, attention_masks = tokenize_data(texts)
    input_ids, val_input_ids, attention_masks, val_attention_masks, labels, val_labels = train_test_split(
        input_ids, attention_masks, labels, test_size=0.2
    )
    model = TopicModel()
    train_model(model, input_ids, attention_masks, labels)
    print('Training complete.')

if __name__ == '__main__':
    main()