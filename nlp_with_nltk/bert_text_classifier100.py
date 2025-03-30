import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

class TextDataset(Dataset):
    """Custom dataset for loading text data."""
    def __init__(self, texts):
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

class BertTextClassifier(nn.Module):
    """BERT-based text classifier."""
    def __init__(self):
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

def train_model(model, dataloader, epochs=3):
    """Trains the BERT text classifier on provided data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, torch.randint(0, 2, (input_ids.size(0),)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

if __name__ == '__main__':
    sample_texts = ["I love programming!", "Deep learning is fascinating.", "Natural language processing is a key area of AI."]
    dataset = TextDataset(sample_texts)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = BertTextClassifier()
    train_model(model, dataloader)