import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class SimpleBertClassifier(nn.Module):
    """A simple classifier using BERT for text classification."""
    def __init__(self, num_classes):
        super(SimpleBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through BERT and the classifier."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def preprocess_data(texts, tokenizer, max_length):
    """Tokenizes and prepares input data for BERT model."""
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings['input_ids'], encodings['attention_mask']

def train_model(model, dataloader, optimizer, device, epochs):
    """Trains the BERT classifier on the provided data."""
    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SimpleBertClassifier(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    texts = ['I love programming.', 'I hate bugs.']
    labels = torch.tensor([1, 0]).unsqueeze(1)
    input_ids, attention_mask = preprocess_data(texts, tokenizer, max_length=10)
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    train_model(model, dataloader, optimizer, device, epochs=3)
    print('Training complete.')