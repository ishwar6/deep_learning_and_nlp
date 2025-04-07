import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SimpleBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_output)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mock_texts = ['Hello, world!', 'Deep learning is fun!']
    mock_labels = torch.tensor([0, 1])
    inputs = tokenizer(mock_texts, padding=True, truncation=True, return_tensors='pt')
    train_loader = [(inputs['input_ids'], inputs['attention_mask'], mock_labels)]
    model = SimpleBERTClassifier(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train_model(model, train_loader, criterion, optimizer, num_epochs=3)

if __name__ == '__main__':
    main()