import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

def load_data(sentences):
    """Tokenizes input sentences and returns input IDs and attention masks."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    return encoding['input_ids'], encoding['attention_mask']

class SentimentModel(nn.Module):
    """Defines a simple sentiment analysis model using BERT."""
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        output = self.dropout(pooled_output)
        return self.fc(output)

def train_model(model, input_ids, attention_mask, labels, epochs=3, lr=1e-5):
    """Trains the sentiment model on the provided data."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    sentences = ["I love this product!", "This is the worst experience I ever had."]
    labels = torch.tensor([1, 0])
    input_ids, attention_mask = load_data(sentences)
    model = SentimentModel()
    train_model(model, input_ids, attention_mask, labels)