import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class TextDataset(Dataset):
    """Custom dataset for text data with tokenization."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoded['input_ids'].squeeze(), 'attention_mask': encoded['attention_mask'].squeeze(), 'label': label}

class SimpleBERTClassifier(nn.Module):
    """Simple BERT-based classifier for text classification."""
    def __init__(self):
        super(SimpleBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

def train_model(model, dataset, epochs=3, batch_size=8):
    """Train the BERT model on the provided dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = loss_fn(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')

if __name__ == '__main__':
    sample_texts = ['I love programming!', 'Deep learning is fascinating.']
    sample_labels = [1, 1]
    dataset = TextDataset(sample_texts, sample_labels)
    model = SimpleBERTClassifier()
    train_model(model, dataset)