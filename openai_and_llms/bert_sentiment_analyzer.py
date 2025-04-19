import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SentimentAnalyzer(nn.Module):
    """
    A sentiment analysis model that uses BERT for text classification.
    """
    def __init__(self):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

def preprocess_data(texts):
    """
    Tokenizes and encodes text data for BERT input.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

def train_model(model, data, labels, epochs=3, lr=5e-5):
    """
    Trains the sentiment analysis model using the provided data and labels.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        input_ids, attention_mask = data
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    sample_texts = ["I love this product!", "This is the worst experience I've ever had."]
    sample_labels = torch.tensor([1, 0])
    input_ids, attention_mask = preprocess_data(sample_texts)
    model = SentimentAnalyzer()
    train_model(model, (input_ids, attention_mask), sample_labels)