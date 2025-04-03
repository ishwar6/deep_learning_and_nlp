import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class TextDataset(Dataset):
    """Custom dataset for text classification using BERT."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label

def train_model(model, data_loader, optimizer, device):
    """Train the BERT model for one epoch."""
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in data_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader)

def main():
    """Main function to execute training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    texts = ['I love programming', 'Python is great', 'I hate bugs', 'Debugging is fun']
    labels = [1, 1, 0, 1]
    dataset = TextDataset(texts, labels, tokenizer, max_len=32)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    avg_loss = train_model(model, data_loader, optimizer, device)
    print(f'Average training loss: {avg_loss:.4f}')

if __name__ == '__main__':
    main()