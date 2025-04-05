import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SentimentAnalyzer(nn.Module):
    """A simple sentiment analysis model using BERT."""
    def __init__(self):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        """Forward pass for the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return self.sigmoid(logits)

def preprocess_data(texts):
    """Tokenizes and prepares input data for BERT."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

def train_model(model, data, labels, epochs=3):
    """Trains the sentiment analysis model."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs, masks = data
        outputs = model(inputs, masks)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

texts = ["I love this product!", "This is the worst experience ever."]
labels = torch.tensor([1, 0])
input_ids, attention_mask = preprocess_data(texts)
model = SentimentAnalyzer()
train_model(model, (input_ids, attention_mask), labels)