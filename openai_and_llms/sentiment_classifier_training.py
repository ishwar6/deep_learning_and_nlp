import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SentimentClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def train_model(model, dataloader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    example_texts = ['I love this product!', 'This is the worst thing I have ever bought.']
    labels = torch.tensor([1, 0])
    inputs = tokenizer(example_texts, padding=True, truncation=True, return_tensors='pt')

    model = SentimentClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    train_model(model, [(inputs['input_ids'], inputs['attention_mask'], labels)], optimizer, criterion)

if __name__ == '__main__':
    main()