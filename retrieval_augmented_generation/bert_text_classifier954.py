import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    """Custom dataset for loading text data for BERT model."""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return { 'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(self.labels[idx], dtype=torch.long) }

class TextClassifier(nn.Module):
    """BERT-based text classification model."""
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids, attention_mask=attention_mask).logits

def train_model(model, dataloader, optimizer, device):
    """Train the BERT text classification model."""
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == '__main__':
    texts = ["I love programming.", "Deep learning is fascinating.", "I dislike bugs.", "Artificial intelligence is the future."]
    labels = [1, 1, 0, 1]
    dataset = TextDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    avg_loss = train_model(model, dataloader, optimizer, device)
    print(f'Average loss: {avg_loss:.4f}')